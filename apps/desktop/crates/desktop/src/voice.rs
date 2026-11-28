//! Native Voice & Audio: intentional on-device dictation and installed system speech.
//! No recognition authorization, audio engine, or microphone is opened at startup.
use super::{Hermes, Message, Session, ui};
use crate::appearance::text;
use iced::{
    Alignment, Element, Fill, Task,
    widget::{Space, button, checkbox, column, container, pick_list, row, scrollable, text_input},
};
use serde::{Deserialize, Serialize};
use std::{io::Write, time::Duration};
#[cfg(not(target_os = "macos"))]
mod system;

#[derive(Clone)]
pub(super) enum Action {
    Record,
    StopRecord,
    CancelRecord,
    Signal(u64, Signal),
    FinishTimeout(u64),
    Speak(String, String),
    StopSpeaking,
    PollSpeech(u64),
    SelectVoice(Voice),
    Rate(f32),
    AutoRead(bool),
    Language(String),
    Save,
    /// Voice conversation mode: listen, send, read the reply aloud, listen again.
    Conversation(bool),
    /// End of an utterance after a pause (epoch, transcript length when scheduled).
    Silence(u64, usize),
    /// Turn the wake word on or off (saved).
    Wake(bool),
    /// Start (or restart) listening for the wake word when idle.
    WakeArm,
    /// The dedicated wake-word model's control pipe, then its events.
    WakeModel(Result<super::wake_model::Listener, super::wake_model::Event>),
}
impl std::fmt::Debug for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("VoiceAction")
    }
}
#[derive(Debug, Clone)]
pub(super) enum Signal {
    SpeechPermission(bool),
    MicrophonePermission(bool),
    Transcript(String, bool),
    Failure(String),
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Voice {
    pub id: String,
    pub name: String,
}
impl std::fmt::Display for Voice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub(super) struct Preferences {
    #[serde(default)]
    pub voice_id: String,
    #[serde(default = "default_speed")]
    pub speed: f32,
    #[serde(default)]
    pub auto_read: bool,
    #[serde(default)]
    pub locale: String,
    /// Listen on-device for “hey Hermes” and start a voice conversation.
    #[serde(default)]
    pub wake_word: bool,
}
fn default_speed() -> f32 {
    1.0
}
impl Default for Preferences {
    fn default() -> Self {
        Self {
            voice_id: String::new(),
            speed: 1.0,
            auto_read: false,
            locale: String::new(),
            wake_word: false,
        }
    }
}

pub(super) struct State {
    pub preferences: Preferences,
    pub voices: Vec<Voice>,
    pub selected_voice: Option<Voice>,
    pub recording: bool,
    pub authorizing: bool,
    pub finishing: bool,
    pub transcript: String,
    pub error: Option<String>,
    pub speaking: Option<String>,
    /// Voice conversation mode is on.
    pub conversation: bool,
    /// The microphone is open only to spot the wake word.
    pub wake_listening: bool,
    /// The running "hey hermes" model (Hermes Python agent), when there is one.
    wake_model: Option<super::wake_model::Listener>,
    wake_model_starting: bool,
    /// Why the dedicated model cannot run here; the phrase-in-speech fallback is used instead.
    wake_model_failed: Option<String>,
    epoch: u64,
    speech_epoch: u64,
    sender: async_channel::Sender<(u64, Signal)>,
    receiver: async_channel::Receiver<(u64, Signal)>,
    #[cfg(target_os = "macos")]
    input: Option<mac::Recording>,
    #[cfg(target_os = "macos")]
    output: Option<mac::Output>,
    #[cfg(not(target_os = "macos"))]
    output: Option<system::Output>,
}
impl Default for State {
    fn default() -> Self {
        let (sender, receiver) = async_channel::unbounded();
        Self {
            preferences: Preferences::default(),
            voices: Vec::new(),
            selected_voice: None,
            recording: false,
            authorizing: false,
            finishing: false,
            transcript: String::new(),
            error: None,
            speaking: None,
            conversation: false,
            wake_listening: false,
            wake_model: None,
            wake_model_starting: false,
            wake_model_failed: None,
            epoch: 0,
            speech_epoch: 0,
            sender,
            receiver,
            #[cfg(target_os = "macos")]
            input: None,
            output: None,
        }
    }
}
impl State {
    pub fn open() -> Self {
        let mut state = Self::default();
        if let Ok(dir) = Session::default_dir() {
            match std::fs::read(dir.join("voice-settings.json")) {
                Ok(bytes) => match serde_json::from_slice::<Preferences>(&bytes) {
                    Ok(p)
                        if p.speed.is_finite()
                            && (0.5..=2.0).contains(&p.speed)
                            && p.locale.len() <= 64 =>
                    {
                        state.preferences = p
                    }
                    Ok(_) | Err(_) => {
                        state.error = Some("Saved voice settings could not be read.".into())
                    }
                },
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => state.error = Some(error.to_string()),
            }
        }
        #[cfg(target_os = "macos")]
        {
            state.voices = mac::voices();
        }
        #[cfg(not(target_os = "macos"))]
        {
            state.voices = system::voices();
        }
        state.selected_voice = state
            .voices
            .iter()
            .find(|voice| voice.id == state.preferences.voice_id)
            .cloned();
        if state.selected_voice.is_none() && !state.preferences.voice_id.is_empty() {
            state.error=Some("Your saved speech voice is not installed. Choose an installed voice to use speech output.".into());
        }
        state
    }
    pub fn events(&self) -> async_channel::Receiver<(u64, Signal)> {
        self.receiver.clone()
    }
    fn save(&self) -> Result<(), String> {
        let dir = Session::default_dir().map_err(|e| e.to_string())?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join("voice-settings.json");
        if std::fs::symlink_metadata(&path).is_ok_and(|meta| meta.file_type().is_symlink()) {
            return Err("Voice settings must not be a link".into());
        }
        let mut file = tempfile::NamedTempFile::new_in(dir).map_err(|e| e.to_string())?;
        serde_json::to_writer_pretty(&mut file, &self.preferences).map_err(|e| e.to_string())?;
        file.flush().map_err(|e| e.to_string())?;
        file.as_file().sync_all().map_err(|e| e.to_string())?;
        file.persist(path).map_err(|e| e.error.to_string())?;
        Ok(())
    }
    pub fn stop_all(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        self.recording = false;
        self.authorizing = false;
        self.finishing = false;
        #[cfg(target_os = "macos")]
        {
            self.input.take();
        }
        if let Some(output) = &self.output {
            output.stop();
        }
        self.speaking = None;
        self.speech_epoch = self.speech_epoch.wrapping_add(1);
    }
    pub fn dictation_unavailable(&self) -> Option<&'static str> {
        #[cfg(target_os = "macos")]
        {
            mac::dictation_unavailable()
        }
        #[cfg(not(target_os = "macos"))]
        {
            Some(
                "On-device dictation is available on macOS 14 or later. This platform does not yet have a native dictation integration.",
            )
        }
    }
    pub fn speech_unavailable(&self) -> Option<&'static str> {
        if !self.voices.is_empty() {
            return None;
        }
        #[cfg(target_os = "windows")]
        {
            Some(
                "No installed Windows System.Speech voice is available. Install a Windows speech voice and restart Hermes.",
            )
        }
        #[cfg(target_os = "linux")]
        {
            Some(
                "Speech output requires eSpeak NG or eSpeak. Install one using your system package manager, then restart Hermes.",
            )
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            Some("No installed system speech voice is available.")
        }
    }
}

impl Hermes {
    pub(super) fn voice_update(&mut self, action: Action) -> Task<Message> {
        match action {
            Action::Record => {
                // Wake listening yields the mic to dictation the user asked for.
                if self.voice.wake_listening && !self.voice.conversation {
                    self.voice.wake_listening = false;
                    let _ = self.voice_update(Action::CancelRecord);
                }
                if let Some(model) = self.voice.wake_model.clone() { let pause = Task::future(async move { model.send("PAUSE").await }).discard(); let record = self.voice_update_record(); return Task::batch([pause, record]); }
                return self.voice_update_record();
            }
            Action::WakeModel(Ok(listener)) => self.voice.wake_model = Some(listener),
            Action::WakeModel(Err(event)) => {
                use super::wake_model::Event;
                match event {
                    Event::Installing => self.note("Installing the “hey Hermes” wake-word model (one time)…"),
                    Event::Ready => { self.voice.wake_model_starting = false; self.note("Wake word on · say “hey Hermes”."); }
                    Event::Wake if self.voice.wake_listening && self.voice.preferences.wake_word => {
                        self.voice.wake_listening = false;
                        return self.voice_update(Action::Conversation(true));
                    }
                    Event::Wake => {}
                    Event::Unavailable(reason) => {
                        self.voice.wake_model_failed = Some(reason);
                        self.voice.wake_model = None; self.voice.wake_model_starting = false; self.voice.wake_listening = false;
                        return self.voice_update(Action::WakeArm);
                    }
                    Event::Exited => { self.voice.wake_model = None; self.voice.wake_model_starting = false; self.voice.wake_listening = false; }
                }
            }
            _ => return self.voice_update_rest(action),
        }
        Task::none()
    }
    fn voice_update_record(&mut self) -> Task<Message> {
        {
            {
                if self.voice.recording {
                    return self.voice_update(Action::StopRecord);
                }
                if self.voice.authorizing || self.voice.finishing {
                    return Task::none();
                }
                if let Some(reason) = self.voice.dictation_unavailable() {
                    self.voice.error = Some(reason.into());
                    return Task::none();
                }
                if let Some(output) = &self.voice.output {
                    output.stop();
                }
                self.voice.speaking = None;
                self.voice.speech_epoch = self.voice.speech_epoch.wrapping_add(1);
                self.voice.error = None;
                self.voice.transcript.clear();
                self.voice.epoch = self.voice.epoch.wrapping_add(1);
                let epoch = self.voice.epoch;
                #[cfg(target_os = "macos")]
                {
                    match mac::authorize_speech(epoch, self.voice.sender.clone()) {
                        Ok(()) => self.voice.authorizing = true,
                        Err(error) => self.voice.error = Some(error),
                    }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = epoch;
                    self.voice.error = Some("Native dictation is available on macOS.".into());
                }
            }
        }
        Task::none()
    }
    fn voice_update_rest(&mut self, action: Action) -> Task<Message> {
        match action {
            Action::Wake(on) => {
                self.voice.preferences.wake_word = on;
                if let Err(e) = self.voice.save() { self.voice.error = Some(e.to_string()); }
                if on { self.note("Wake word on · say “hey Hermes”. Listening happens on this Mac."); return self.voice_update(Action::WakeArm); }
                self.note("Wake word off.");
                if let Some(model) = self.voice.wake_model.take() { self.voice.wake_listening = false; return Task::future(async move { model.send("STOP").await }).discard(); }
                if self.voice.wake_listening { self.voice.wake_listening = false; return self.voice_update(Action::CancelRecord); }
            }
            Action::WakeArm => {
                let idle = !self.voice.recording && !self.voice.authorizing && !self.voice.finishing && !self.voice.conversation && self.voice.speaking.is_none();
                if !self.voice.preferences.wake_word || !idle || self.voice.wake_model_starting { return Task::none() }
                // The dedicated model first (Electron's detector); phrase-in-speech when it cannot run.
                if let Some(model) = self.voice.wake_model.clone() {
                    self.voice.wake_listening = true;
                    return Task::future(async move { model.send("RESUME").await }).discard();
                }
                if self.voice.wake_model_failed.is_none() && super::wake_model::available() {
                    self.voice.wake_model_starting = true;
                    self.voice.wake_listening = true;
                    let data = self.library.as_ref().map(|l| l.root().to_owned()).unwrap_or_else(std::env::temp_dir);
                    return Task::run(super::wake_model::start(data), |event| Message::Voice(Action::WakeModel(event)));
                }
                if self.voice.dictation_unavailable().is_none() {
                    let record = self.voice_update_record();
                    // Set after Record: Record itself treats a set flag as "yield the mic".
                    self.voice.wake_listening = self.voice.recording || self.voice.authorizing;
                    return record;
                }
            }
            Action::Signal(epoch, Signal::Transcript(text, final_result)) if epoch == self.voice.epoch && self.voice.wake_listening => {
                if wake_phrase(&text) {
                    self.voice.wake_listening = false;
                    let _ = self.voice_update(Action::CancelRecord);
                    return self.voice_update(Action::Conversation(true));
                }
                if final_result || text.len() > 400 {
                    // A recognition session ends after about a minute: start a fresh one.
                    let _ = self.voice_update(Action::CancelRecord);
                    self.voice.wake_listening = false;
                    return Task::perform(async { tokio::time::sleep(Duration::from_millis(300)).await; }, |_| Message::Voice(Action::WakeArm));
                }
            }
            Action::Signal(epoch, Signal::Failure(_)) if epoch == self.voice.epoch && self.voice.wake_listening => {
                let _ = self.voice_update(Action::CancelRecord);
                self.voice.wake_listening = false;
                return Task::perform(async { tokio::time::sleep(Duration::from_secs(3)).await; }, |_| Message::Voice(Action::WakeArm));
            }
            Action::Signal(epoch, signal) if epoch == self.voice.epoch => match signal {
                Signal::SpeechPermission(true) if self.voice.authorizing => {
                    #[cfg(target_os = "macos")]
                    if let Err(error) = mac::authorize_microphone(epoch, self.voice.sender.clone())
                    {
                        self.voice.authorizing = false;
                        self.voice.error = Some(error);
                    }
                }
                Signal::SpeechPermission(false) => {
                    self.voice.authorizing = false;
                    self.voice.error=Some("Speech recognition permission was denied. You can allow Hermes in System Settings → Privacy & Security → Speech Recognition.".into());
                }
                Signal::MicrophonePermission(true) if self.voice.authorizing => {
                    self.voice.authorizing = false;
                    #[cfg(target_os = "macos")]
                    {
                        match mac::Recording::start(
                            &self.voice.preferences.locale,
                            epoch,
                            self.voice.sender.clone(),
                        ) {
                            Ok(input) => {
                                self.voice.input = Some(input);
                                self.voice.recording = true;
                            }
                            Err(error) => self.voice.error = Some(error),
                        }
                    }
                }
                Signal::MicrophonePermission(false) => {
                    self.voice.authorizing = false;
                    self.voice.error=Some("Microphone permission was denied. You can allow Hermes in System Settings → Privacy & Security → Microphone.".into());
                }
                Signal::Transcript(text, final_result) => {
                    self.voice.transcript = text;
                    if final_result {
                        self.finish_dictation();
                        return self.conversation_send();
                    }
                    // In a voice conversation a 1.5s pause ends the utterance.
                    if self.voice.conversation && self.voice.recording {
                        let (epoch, len) = (self.voice.epoch, self.voice.transcript.len());
                        return Task::perform(async { tokio::time::sleep(Duration::from_millis(1500)).await; }, move |_| Message::Voice(Action::Silence(epoch, len)));
                    }
                }
                Signal::Failure(error) => {
                    self.voice.error = Some(error);
                    self.finish_dictation();
                }
                _ => {}
            },
            Action::StopRecord if self.voice.recording => {
                self.voice.recording = false;
                self.voice.finishing = true;
                #[cfg(target_os = "macos")]
                if let Some(input) = &mut self.voice.input {
                    input.finish();
                }
                let epoch = self.voice.epoch;
                return Task::perform(
                    async {
                        tokio::time::sleep(Duration::from_secs(3)).await;
                    },
                    move |_| Message::Voice(Action::FinishTimeout(epoch)),
                );
            }
            Action::FinishTimeout(epoch) if epoch == self.voice.epoch && self.voice.finishing => {
                self.finish_dictation();
                return self.conversation_send();
            }
            Action::Silence(epoch, len) if epoch == self.voice.epoch && self.voice.recording && self.voice.transcript.len() == len && len > 0 => {
                return self.voice_update(Action::StopRecord);
            }
            Action::Conversation(on) => {
                if self.voice.wake_listening { self.voice.wake_listening = false; let _ = self.voice_update(Action::CancelRecord); }
                self.voice.conversation = on;
                if on { return self.voice_update(Action::Record); }
                let _ = self.voice_update(Action::CancelRecord);
                let stop = self.voice_update(Action::StopSpeaking);
                return Task::batch([stop, Task::perform(async { tokio::time::sleep(Duration::from_millis(800)).await; }, |_| Message::Voice(Action::WakeArm))]);
            }
            Action::CancelRecord => {
                self.voice.epoch = self.voice.epoch.wrapping_add(1);
                self.voice.recording = false;
                self.voice.finishing = false;
                self.voice.authorizing = false;
                self.voice.transcript.clear();
                #[cfg(target_os = "macos")]
                self.voice.input.take();
            }
            Action::Speak(id, body) => {
                if let Some(reason) = self.voice.speech_unavailable() {
                    self.voice.error = Some(reason.into());
                    return Task::none();
                }
                if self.voice.recording || self.voice.finishing {
                    self.finish_dictation();
                }
                if self.voice.authorizing {
                    let _ = self.voice_update(Action::CancelRecord);
                }
                self.voice.error = None;
                self.voice.speech_epoch = self.voice.speech_epoch.wrapping_add(1);
                #[cfg(target_os = "macos")]
                {
                    let output = self.voice.output.get_or_insert_with(mac::Output::new);
                    match output.speak(&body, &self.voice.preferences) {
                        Ok(()) => {
                            self.voice.speaking = Some(id);
                            let epoch = self.voice.speech_epoch;
                            return Task::perform(
                                async {
                                    tokio::time::sleep(Duration::from_millis(250)).await;
                                },
                                move |_| Message::Voice(Action::PollSpeech(epoch)),
                            );
                        }
                        Err(error) => self.voice.error = Some(error),
                    }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let output = self.voice.output.get_or_insert_with(system::Output::new);
                    match output.speak(&body, &self.voice.preferences) {
                        Ok(()) => {
                            self.voice.speaking = Some(id);
                            let epoch = self.voice.speech_epoch;
                            return Task::perform(
                                async {
                                    tokio::time::sleep(Duration::from_millis(250)).await;
                                },
                                move |_| Message::Voice(Action::PollSpeech(epoch)),
                            );
                        }
                        Err(error) => self.voice.error = Some(error),
                    }
                }
            }
            Action::PollSpeech(epoch) if epoch == self.voice.speech_epoch => {
                #[cfg(target_os = "macos")]
                if self
                    .voice
                    .output
                    .as_ref()
                    .is_some_and(mac::Output::speaking)
                {
                    return Task::perform(
                        async {
                            tokio::time::sleep(Duration::from_millis(250)).await;
                        },
                        move |_| Message::Voice(Action::PollSpeech(epoch)),
                    );
                }
                #[cfg(not(target_os = "macos"))]
                if let Some(output) = &self.voice.output {
                    match output.speaking() {
                        Ok(true) => {
                            return Task::perform(
                                async {
                                    tokio::time::sleep(Duration::from_millis(250)).await;
                                },
                                move |_| Message::Voice(Action::PollSpeech(epoch)),
                            );
                        }
                        Err(error) => self.voice.error = Some(error),
                        Ok(false) => {}
                    }
                }
                self.voice.speaking = None;
                // The reply has been read: listen for the next turn.
                if self.voice.conversation && !self.running { return self.voice_update(Action::Record); }
            }
            Action::StopSpeaking => {
                if let Some(output) = &self.voice.output {
                    output.stop();
                }
                self.voice.speaking = None;
                self.voice.speech_epoch = self.voice.speech_epoch.wrapping_add(1);
            }
            Action::SelectVoice(voice) => {
                self.voice.preferences.voice_id = voice.id.clone();
                self.voice.selected_voice = Some(voice);
                if let Err(error) = self.voice.save() {
                    self.voice.error = Some(error);
                }
            }
            Action::Rate(speed) => {
                self.voice.preferences.speed = speed.clamp(0.5, 2.0);
                if let Err(error) = self.voice.save() {
                    self.voice.error = Some(error);
                }
            }
            Action::AutoRead(value) => {
                self.voice.preferences.auto_read = value;
                if let Err(error) = self.voice.save() {
                    self.voice.error = Some(error);
                }
            }
            Action::Language(value) => {
                if value.len() <= 64 {
                    self.voice.preferences.locale = value;
                }
            }
            Action::Save => match self.voice.save() {
                Ok(()) => self.note("Voice settings saved"),
                Err(error) => self.voice.error = Some(error),
            },
            _ => {}
        }
        Task::none()
    }
    /// In a voice conversation, send what was just dictated.
    fn conversation_send(&mut self) -> Task<Message> {
        if self.voice.conversation && !self.running && !self.composer.text().trim().is_empty() { return self.update(Message::Send); }
        Task::none()
    }
    fn finish_dictation(&mut self) {
        let text = std::mem::take(&mut self.voice.transcript);
        self.voice.epoch = self.voice.epoch.wrapping_add(1);
        self.voice.recording = false;
        self.voice.authorizing = false;
        self.voice.finishing = false;
        #[cfg(target_os = "macos")]
        self.voice.input.take();
        if !text.trim().is_empty() {
            let prefix = if self.composer.text().is_empty() {
                ""
            } else {
                " "
            };
            self.composer
                .perform(iced::widget::text_editor::Action::Edit(
                    iced::widget::text_editor::Edit::Paste(std::sync::Arc::from(format!(
                        "{prefix}{}",
                        text.trim()
                    ))),
                ));
            if let Err(error) = self.save_draft() {
                self.error(format!(
                    "Dictation added, but the draft could not be saved: {error}"
                ));
            }
        }
    }
}

pub(super) fn view(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark);
    let mut body = column![
        super::settings::section_title("Dictation", app.dark)
    ]
    .spacing(12);
    if let Some(reason) = app.voice.dictation_unavailable() {
        body = body.push(text(reason).size(12).color(p.muted));
    } else {
        let language = text_input(
            "System default (or locale, e.g. en-US)",
            &app.voice.preferences.locale,
        )
        .on_input(|value| Message::Voice(Action::Language(value)))
        .size(12)
        .padding(8);
        body=body.push(text("Use the microphone button to dictate into your message draft. Recognition stays on this Mac and requires an installed on-device language model.").size(12).color(p.muted))
            .push(row![text("Language").size(13).width(110),language].spacing(12).align_y(Alignment::Center))
            .push(button(text("Save").size(12)).on_press(Message::Voice(Action::Save)).padding([6,12]));
    }
    body = body
        .push(Space::new().height(18))
        .push(text("Speech output").size(14));
    if let Some(reason) = app.voice.speech_unavailable() {
        body = body.push(text(reason).size(12).color(p.muted));
    } else {
        let voice = pick_list(
            app.voice.voices.clone(),
            app.voice.selected_voice.clone(),
            |voice| Message::Voice(Action::SelectVoice(voice)),
        )
        .placeholder("Choose voice")
        .text_size(12)
        .padding(7);
        let mut rates = row![text("Speed").size(13), Space::new().width(Fill)]
            .spacing(5)
            .align_y(Alignment::Center);
        for rate in [0.75, 1.0, 1.25, 1.5, 2.0] {
            let active = (app.voice.preferences.speed - rate).abs() < 0.01;
            rates = rates.push(
                button(text(format!("{rate}×")).size(11))
                    .on_press(Message::Voice(Action::Rate(rate)))
                    .padding([5, 8])
                    .style(move |theme, status| {
                        if active {
                            button::primary(theme, status)
                        } else {
                            button::text(theme, status)
                        }
                    }),
            );
        }
        body=body.push(row![text("Voice").size(13).width(110),voice].spacing(12).align_y(Alignment::Center)).push(rates)
        .push(checkbox(app.voice.preferences.auto_read).label("Read replies aloud").on_toggle(|value|Message::Voice(Action::AutoRead(value))).text_size(12))
        .push(text("Installed system voices provide speech output. You can also read an individual reply aloud from its message controls.").size(12).color(p.muted));
    }
    if app.voice.speaking.is_some() {
        body = body.push(
            button(text("Stop speaking").size(12))
                .on_press(Message::Voice(Action::StopSpeaking))
                .padding([6, 12])
                .style(button::secondary),
        );
    }
    if let Some(error) = &app.voice.error {
        body = body.push(text(error).size(12).color(p.muted));
    }
    scrollable(container(body).padding([12, 32]).width(Fill))
        .height(Fill)
        .into()
}

#[cfg(target_os = "macos")]
mod mac {
    use super::{Preferences, Signal, Voice};
    use blockmodern::RcBlock;
    use foundation3::{NSBundle, NSError, NSLocale, NSString};
    use objc2_avf_audio::{
        AVAudioApplication, AVAudioEngine, AVAudioInputNode, AVAudioPCMBuffer, AVAudioTime,
        AVSpeechBoundary, AVSpeechSynthesisVoice, AVSpeechSynthesizer, AVSpeechUtterance,
    };
    use objc2_speech::{
        SFSpeechAudioBufferRecognitionRequest, SFSpeechRecognitionResult, SFSpeechRecognitionTask,
        SFSpeechRecognizer, SFSpeechRecognizerAuthorizationStatus,
    };
    use objc2modern::{
        AnyThread,
        rc::Retained,
        runtime::{AnyClass, Bool},
    };
    use std::ptr::NonNull;
    type Sender = async_channel::Sender<(u64, Signal)>;

    pub fn voices() -> Vec<Voice> {
        unsafe {
            let mut voices = vec![Voice {
                id: String::new(),
                name: "System voice".into(),
            }];
            let installed = AVSpeechSynthesisVoice::speechVoices();
            for index in 0..installed.count() {
                let voice = installed.objectAtIndex(index);
                voices.push(Voice {
                    id: voice.identifier().to_string(),
                    name: format!("{} ({})", voice.name(), voice.language()),
                });
            }
            voices
        }
    }
    pub fn authorize_speech(epoch: u64, sender: Sender) -> Result<(), String> {
        if let Some(reason) = dictation_unavailable() {
            return Err(reason.into());
        }
        let block = RcBlock::new(move |status: SFSpeechRecognizerAuthorizationStatus| {
            let _ = sender.try_send((
                epoch,
                Signal::SpeechPermission(
                    status == SFSpeechRecognizerAuthorizationStatus::Authorized,
                ),
            ));
        });
        unsafe {
            SFSpeechRecognizer::requestAuthorization(&block);
        }
        Ok(())
    }
    pub fn dictation_unavailable() -> Option<&'static str> {
        let bundle = NSBundle::mainBundle();
        for key in [
            "NSMicrophoneUsageDescription",
            "NSSpeechRecognitionUsageDescription",
        ] {
            if bundle
                .objectForInfoDictionaryKey(&NSString::from_str(key))
                .is_none()
            {
                return Some(
                    "Dictation requires the packaged Hermes.app with speech and microphone usage descriptions. Launch Hermes.app to use it.",
                );
            }
        }
        if AnyClass::get(c"AVAudioApplication").is_none() {
            return Some(
                "On-device dictation requires macOS 14 or later. Read aloud remains available.",
            );
        }
        None
    }
    pub fn authorize_microphone(epoch: u64, sender: Sender) -> Result<(), String> {
        if AnyClass::get(c"AVAudioApplication").is_none() {
            return Err(
                "Native microphone authorization is unavailable on this macOS version.".into(),
            );
        }
        let block = RcBlock::new(move |granted: Bool| {
            let _ = sender.try_send((epoch, Signal::MicrophonePermission(granted.as_bool())));
        });
        unsafe {
            AVAudioApplication::requestRecordPermissionWithCompletionHandler(&block);
        }
        Ok(())
    }

    pub struct Recording {
        _recognizer: Retained<SFSpeechRecognizer>,
        engine: Retained<AVAudioEngine>,
        input: Retained<AVAudioInputNode>,
        request: Retained<SFSpeechAudioBufferRecognitionRequest>,
        task: Retained<SFSpeechRecognitionTask>,
        tapped: bool,
    }
    impl Recording {
        pub fn start(language: &str, epoch: u64, sender: Sender) -> Result<Self, String> {
            unsafe {
                let recognizer = if language.trim().is_empty() {
                    SFSpeechRecognizer::init(SFSpeechRecognizer::alloc())
                } else {
                    let locale = NSLocale::initWithLocaleIdentifier(
                        NSLocale::alloc(),
                        &NSString::from_str(language.trim()),
                    );
                    SFSpeechRecognizer::initWithLocale(SFSpeechRecognizer::alloc(), &locale)
                }
                .ok_or("Speech recognition is unavailable for that language")?;
                if !recognizer.supportsOnDeviceRecognition() {
                    return Err("No on-device dictation model is available for this language. Enable or download its dictation language in macOS System Settings, then try again. Audio was not sent to a server.".into());
                }
                if !recognizer.isAvailable() {
                    return Err("On-device speech recognition is temporarily unavailable.".into());
                }
                let request = SFSpeechAudioBufferRecognitionRequest::new();
                request.setRequiresOnDeviceRecognition(true);
                request.setShouldReportPartialResults(true);
                let results = RcBlock::new(
                    move |result: *mut SFSpeechRecognitionResult, error: *mut NSError| {
                        if let Some(result) = result.as_ref() {
                            let transcript =
                                result.bestTranscription().formattedString().to_string();
                            let _ = sender.try_send((
                                epoch,
                                Signal::Transcript(transcript, result.isFinal()),
                            ));
                        }
                        if let Some(error) = error.as_ref() {
                            let _ = sender.try_send((
                                epoch,
                                Signal::Failure(error.localizedDescription().to_string()),
                            ));
                        }
                    },
                );
                let task = recognizer.recognitionTaskWithRequest_resultHandler(&request, &results);
                let engine = AVAudioEngine::new();
                let input = engine.inputNode();
                let format = input.outputFormatForBus(0);
                if format.channelCount() == 0 || format.sampleRate() <= 0.0 {
                    task.cancel();
                    return Err("No usable microphone input is available.".into());
                }
                let append = request.clone();
                let tap = RcBlock::new(
                    move |buffer: NonNull<AVAudioPCMBuffer>, _: NonNull<AVAudioTime>| {
                        append.appendAudioPCMBuffer(buffer.as_ref())
                    },
                );
                input.installTapOnBus_bufferSize_format_block(
                    0,
                    1024,
                    Some(&format),
                    RcBlock::as_ptr(&tap),
                );
                let mut recording = Self {
                    _recognizer: recognizer,
                    engine,
                    input,
                    request,
                    task,
                    tapped: true,
                };
                recording.engine.prepare();
                if let Err(error) = recording.engine.startAndReturnError() {
                    recording.finish();
                    return Err(error.localizedDescription().to_string());
                }
                Ok(recording)
            }
        }
        pub fn finish(&mut self) {
            unsafe {
                self.engine.stop();
                if self.tapped {
                    self.input.removeTapOnBus(0);
                    self.tapped = false;
                }
                self.request.endAudio();
                self.task.finish();
            }
        }
    }
    impl Drop for Recording {
        fn drop(&mut self) {
            unsafe {
                self.engine.stop();
                if self.tapped {
                    self.input.removeTapOnBus(0);
                }
                self.request.endAudio();
                self.task.cancel();
            }
        }
    }
    pub struct Output {
        synth: Retained<AVSpeechSynthesizer>,
    }
    impl Output {
        pub fn new() -> Self {
            Self {
                synth: unsafe { AVSpeechSynthesizer::new() },
            }
        }
        pub fn speak(&self, text: &str, preferences: &Preferences) -> Result<(), String> {
            unsafe {
                self.stop();
                if text.trim().is_empty() {
                    return Err("This message has no text to read.".into());
                }
                let utterance =
                    AVSpeechUtterance::speechUtteranceWithString(&NSString::from_str(text));
                if !preferences.voice_id.is_empty() {
                    let voice = AVSpeechSynthesisVoice::voiceWithIdentifier(&NSString::from_str(
                        &preferences.voice_id,
                    ))
                    .ok_or("The selected speech voice is no longer installed")?;
                    utterance.setVoice(Some(&voice));
                }
                utterance.setRate((0.5 * preferences.speed).clamp(0.1, 1.0));
                self.synth.speakUtterance(&utterance);
                Ok(())
            }
        }
        pub fn speaking(&self) -> bool {
            unsafe { self.synth.isSpeaking() }
        }
        pub fn stop(&self) {
            unsafe {
                self.synth
                    .stopSpeakingAtBoundary(AVSpeechBoundary::Immediate);
            }
        }
    }
    impl Drop for Output {
        fn drop(&mut self) {
            self.stop();
        }
    }
}

/// Whether a live transcript contains the wake phrase (and its common mis-hearings).
fn wake_phrase(text: &str) -> bool {
    let t: String = text.to_lowercase().chars().map(|c| if c.is_alphanumeric() || c == ' ' { c } else { ' ' }).collect();
    let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    ["hey hermes", "hi hermes", "hey hermès", "hey herme s", "hay hermes", "a hermes", "hey her mees"].iter().any(|p| t.contains(p))
}

#[cfg(test)]
mod wake_tests {
    #[test]
    fn spots_the_wake_phrase() {
        assert!(super::wake_phrase("Hey, Hermes! What's up"));
        assert!(super::wake_phrase("ok so hey hermes"));
        assert!(!super::wake_phrase("Hermes is a Greek god"));
        assert!(!super::wake_phrase("hey there"));
    }
}
