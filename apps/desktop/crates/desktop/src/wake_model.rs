//! The dedicated wake-word model, as Electron runs it: Hermes's own detector
//! (tools/wake_word.py, the bundled "hey hermes" openWakeWord model) in the
//! installed Hermes Python agent, here through assets/wake_listener.py. The
//! listener owns the microphone only while armed: it is paused before
//! dictation opens the mic and resumed when the voice conversation ends.
//! Without the Python agent the desktop falls back to spotting the phrase in
//! live speech recognition (voice.rs).
use std::{process::Stdio, sync::Arc};
use tokio::{io::{AsyncBufReadExt, AsyncWriteExt, BufReader}, process::ChildStdin, sync::Mutex};

const SCRIPT: &str = include_str!("../assets/wake_listener.py");

#[derive(Debug, Clone)]
pub(super) enum Event { Installing, Ready, Wake, Unavailable(String), Exited }

/// The running listener's control pipe; dropping the last copy stops it.
#[derive(Clone)]
pub(super) struct Listener { stdin: Arc<Mutex<ChildStdin>> }
impl std::fmt::Debug for Listener { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("Listener") } }

impl Listener {
    pub async fn send(&self, command: &'static str) {
        let mut pipe = self.stdin.lock().await;
        let _ = pipe.write_all(format!("{command}\n").as_bytes()).await;
        let _ = pipe.flush().await;
    }
}

/// Whether the dedicated model can run here (the Hermes Python agent is installed).
pub(super) fn available() -> bool { hermes_engine::plugins::python_hermes().is_some() }

/// Start the listener; the stream yields the control pipe first, then its events.
pub(super) fn start(data: std::path::PathBuf) -> impl iced::futures::Stream<Item = Result<Listener, Event>> {
    iced::stream::channel(16, move |mut out: iced::futures::channel::mpsc::Sender<Result<Listener, Event>>| async move {
        use iced::futures::SinkExt;
        let Some((python, agent)) = hermes_engine::plugins::python_hermes() else { let _ = out.send(Err(Event::Unavailable("The Hermes agent is not installed".into()))).await; return };
        let script = data.join("wake_listener.py");
        if std::fs::read_to_string(&script).ok().as_deref() != Some(SCRIPT) && let Err(e) = std::fs::write(&script, SCRIPT) {
            let _ = out.send(Err(Event::Unavailable(e.to_string()))).await; return;
        }
        let child = tokio::process::Command::new(python).arg("-I").arg(&script).arg(&agent)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).kill_on_drop(true).spawn();
        let mut child = match child { Ok(c) => c, Err(e) => { let _ = out.send(Err(Event::Unavailable(e.to_string()))).await; return } };
        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else { return };
        let listener = Listener { stdin: Arc::new(Mutex::new(stdin)) };
        let _ = out.send(Ok(listener)).await;
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let event = match line.trim() {
                "READY" => Event::Ready,
                "WAKE" => Event::Wake,
                "INSTALLING" => Event::Installing,
                other => match other.strip_prefix("UNAVAILABLE ") { Some(reason) => Event::Unavailable(reason.to_owned()), None => continue },
            };
            if out.send(Err(event)).await.is_err() { break }
        }
        let _ = child.wait().await;
        let _ = out.send(Err(Event::Exited)).await;
    })
}

#[cfg(test)]
mod tests {
    use iced::futures::StreamExt;
    /// Needs the Hermes Python agent and a microphone: `cargo test -p hermes-desktop -- --ignored wake_model`
    #[tokio::test]
    #[ignore = "opens the microphone through the installed Hermes agent"]
    async fn the_hey_hermes_model_starts_and_stops() {
        let dir = tempfile::tempdir().unwrap();
        let mut stream = Box::pin(super::start(dir.path().to_owned()));
        let listener = match stream.next().await { Some(Ok(l)) => l, other => panic!("no control pipe: {other:?}") };
        let ready = tokio::time::timeout(std::time::Duration::from_secs(240), async {
            while let Some(event) = stream.next().await {
                match event { Err(super::Event::Ready) => return Ok(()), Err(super::Event::Unavailable(r)) => return Err(r), _ => {} }
            }
            Err("exited".into())
        }).await.expect("ready in time");
        assert_eq!(ready, Ok(()));
        listener.send("PAUSE").await;
        listener.send("RESUME").await;
        listener.send("STOP").await;
        let exited = tokio::time::timeout(std::time::Duration::from_secs(20), async { while let Some(e) = stream.next().await { if matches!(e, Err(super::Event::Exited)) { return true } } false }).await;
        assert_eq!(exited, Ok(true), "STOP ends the listener");
    }
}
