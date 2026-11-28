//! Slash commands (lib/desktop-slash-commands.ts, composer/trigger-popover.tsx).
//! Typing "/" opens a completion list above the composer: desktop commands and
//! installed skills, filtered as you type. Sending a command runs it here;
//! "/<skill> …" sends the prompt with that skill loaded.
use super::{Hermes, Message, Screen, models, reasoning, settings, ui};
use crate::appearance::text;
use iced::{Alignment, Color, Element, Fill, Task, widget::{Space, button, column, container, row, scrollable, text_editor}};

/// (name, aliases, description, takes an argument)
/// The desktop-visible commands of desktop-slash-registry.json with the upstream descriptions.
const COMMANDS: &[(&str, &[&str], &str, bool)] = &[
    ("/agents", &["/tasks"], "Show active agents and running tasks", false),
    ("/approvals", &[], "Show or set the persistent dangerous-command approval mode", true),
    ("/battery", &[], "Toggle a color-coded battery indicator in the status bar", false),
    ("/bg", &["/background"], "Run a prompt in a separate background session", true),
    ("/blueprint", &["/bp"], "Set up an automation from a blueprint template", false),
    ("/branch", &["/fork"], "Branch the latest message into a new chat", false),
    ("/browser", &[], "Manage the agent browser [connect|disconnect|status]", false),
    ("/btw", &[], "Ask a side question about this conversation without interrupting it", true),
    ("/bundles", &[], "List skill bundles (aliases /<name> for multiple skills)", false),
    ("/codex-runtime", &["/codex_runtime"], "Toggle codex app-server runtime for OpenAI/Codex models", false),
    ("/compress", &["/compact"], "Compress this conversation context", false),
    ("/config", &[], "Show current configuration", false),
    ("/context", &["/ctx"], "Show detailed context window view with usage gauge, category breakdown, compression stats, and throughput", false),
    ("/debug", &[], "Debug report (versions, config keys, error counts; no log text): shows the privacy notice; /debug upload shares links", false),
    ("/diff", &[], "Show git changes in the working directory", false),
    ("/egress", &[], "Show Docker egress proxy status", false),
    ("/export", &[], "Export a profile (config, skills, theme) to a shareable archive", false),
    ("/focus", &[], "Toggle focus view — show only your prompt and the final response", false),
    ("/goal", &[], "Set a standing goal Hermes works on across turns until achieved", true),
    ("/handoff", &[], "Hand off this session to a messaging platform", true),
    ("/hatch", &["/generate-pet"], "Generate a new pet (opens the pet generator)", false),
    ("/heartbeat", &["/hb"], "Set a recurring prompt that re-enters this session when idle", true),
    ("/help", &["/commands"], "Show desktop slash commands", false),
    ("/import", &[], "Import a shared profile archive as a new profile", false),
    ("/init", &[], "Generate or update AGENTS.md project instructions from a repo scan", false),
    ("/journey", &["/learning", "/memory-graph"], "Open the memory graph — skills + memories over time", false),
    ("/learn", &[], "Learn a reusable skill from anything you describe (dirs, URLs, this chat, notes)", true),
    ("/loop", &[], "Re-run a prompt on a recurring interval in this session", true),
    ("/memory", &[], "Review pending memory writes / toggle the approval gate", false),
    ("/moa", &[], "Run one prompt through the default Mixture of Agents preset, then restore your model", false),
    ("/new", &["/reset"], "Start a new desktop chat", false),
    ("/palette", &[], "Open the fuzzy command palette (also Ctrl+P)", false),
    ("/pause", &[], "Pause new work globally (emergency stop); '/pause off' resumes", false),
    ("/personality", &[], "Set a predefined personality", true),
    ("/pet", &[], "Toggle or adopt a petdex mascot (/pet, /pet list, /pet boba)", false),
    ("/plan", &[], "Write a markdown implementation plan to .hermes/plans/ without executing anything", true),
    ("/platform", &[], "Pause, resume, or list a failing gateway platform", false),
    ("/profile", &[], "Switch the active Hermes profile", true),
    ("/prompt", &[], "Compose your next prompt in $EDITOR (markdown), then send it", false),
    ("/queue", &["/q"], "Queue a prompt for the next turn, or list/edit/rm/move/clear queued prompts", true),
    ("/reasoning", &[], "Reasoning effort or display [<level> [--global]|show|hide|full|clamp]", true),
    ("/refine", &[], "Review this conversation now and save lessons to memory/skills", false),
    ("/resume", &["/sessions", "/switch"], "Resume a saved session", false),
    ("/retry", &[], "Retry the last message (resend to agent)", false),
    ("/review", &[], "Spawn an independent subagent to review the work just discussed (PR, code, docs)", false),
    ("/rollback", &[], "List or restore filesystem checkpoints (restores keep your hand-edits; --all overrides)", false),
    ("/save", &[], "Save the current transcript to JSON", true),
    ("/skills", &[], "Review staged skill writes and approval mode", false),
    ("/skin", &[], "Switch desktop theme or cycle to the next one", true),
    ("/start", &[], "Acknowledge platform start pings without a reply", false),
    ("/status", &[], "Show current session status", false),
    ("/steer", &["/s"], "Inject a message after the next tool call without interrupting", true),
    ("/stop", &[], "Stop the active turn and background processes", false),
    ("/subgoal", &[], "Add or manage extra criteria on the active goal", true),
    ("/subscription", &[], "View your Nous plan and change it in the browser", false),
    ("/suggestions", &[], "Review suggested automations (accept/dismiss)", false),
    ("/timestamps", &["/ts"], "Toggle [HH:MM] timestamps on messages and /history", false),
    ("/title", &[], "Rename the current session", true),
    ("/tools", &[], "Manage tools: /tools [list|disable|enable] [name...]", true),
    ("/topic", &[], "Enable or inspect Telegram DM topic sessions", false),
    ("/topup", &[], "Show your Nous balance and manage billing on the portal", false),
    ("/undo", &[], "Back up N user turns and re-prompt (default 1)", true),
    ("/usage", &[], "Show token usage and rate limits; `reset` redeems a banked Codex limit reset", false),
    ("/version", &["/v"], "Show Hermes Agent version", false),
    ("/wake", &[], "Control the desktop wake-word listener [on|off|status]", true),
    ("/whoami", &[], "Show your slash command access (admin / user)", false),
    ("/yolo", &[], "Toggle YOLO — auto-approve dangerous commands", false),
    ("/model", &[], "Switch the model for this session", false),
];
/// Registry entries marked "terminal": they belong to the Hermes CLI.
const TERMINAL_ONLY: &[&str] = &["/busy", "/clear", "/copy", "/cron", "/exit", "/footer", "/gateway", "/image", "/indicator", "/paste", "/platforms", "/plugins", "/quit", "/redraw", "/reload", "/restart", "/rewind", "/sb", "/set-home", "/sethome", "/snap", "/snapshot", "/statusbar", "/toolsets", "/update", "/verbose", "/worktree"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Item { pub name: String, pub description: String, pub group: &'static str }

#[derive(Default)]
pub(super) struct State { pub index: usize, pub dismissed_for: Option<String> }

/// The completion list for the composer's text, or empty when no popup should show.
pub(super) fn items(app: &Hermes) -> Vec<Item> {
    let text = app.composer.text();
    let line = text.trim_end_matches('\n');
    if app.slash.dismissed_for.as_deref() != Some(line) && let Some(refs) = reference_items(app, line) { return refs; }
    if !line.starts_with('/') || line.contains(char::is_whitespace) || app.slash.dismissed_for.as_deref() == Some(line) { return vec![]; }
    let query = line[1..].to_lowercase();
    let mut out: Vec<Item> = COMMANDS.iter().filter(|(name, _, _, _)| *name != "/model").filter(|(name, aliases, _, _)| name[1..].starts_with(&query) || aliases.iter().any(|a| a[1..].starts_with(&query)))
        .map(|(name, _, description, _)| Item { name: (*name).into(), description: (*description).into(), group: "Commands" }).collect();
    if let Some(library) = &app.library && let Ok(status) = hermes_engine::skill_catalog::status(library.root()) {
        let mut skills: Vec<Item> = status.installed.iter().filter(|id| !status.disabled.contains(*id))
            .filter_map(|id| { let slug = id.rsplit('/').next()?; slug.to_lowercase().contains(&query).then(|| Item { name: format!("/{slug}"), description: "Skill".into(), group: "Skills" }) })
            .collect();
        skills.sort_by(|a, b| a.name.cmp(&b.name)); skills.truncate(20);
        out.extend(skills);
    }
    out
}

/// The `@` token being typed at the end of the composer, if any.
fn reference_token(line: &str) -> Option<&str> {
    let token = line.rsplit(char::is_whitespace).next()?;
    (token.starts_with('@') && line.len() >= token.len() && !line.ends_with(char::is_whitespace)).then_some(token)
}

/// `@` completions (app/chat/composer: kinds, then workspace paths for @file:/@folder:).
fn reference_items(app: &Hermes, line: &str) -> Option<Vec<Item>> {
    let token = reference_token(line)?;
    let item = |name: String, description: &str| Item { name, description: description.into(), group: "References" };
    let Some((kind, partial)) = token[1..].split_once(':') else {
        let query = token[1..].to_lowercase();
        let kinds = [("@file:", "Attach a file's contents"), ("@folder:", "Attach a folder listing"), ("@diff", "Unstaged git changes"), ("@staged", "Staged git changes"),
            ("@git:1", "Recent commits with patches (1–10)"), ("@url:", "Fetch a web page's text")];
        return Some(kinds.iter().filter(|(k, _)| k[1..].starts_with(&query)).map(|(k, d)| item((*k).into(), d)).collect());
    };
    if !matches!(kind, "file" | "folder") { return None; }
    let root = app.workspace()?;
    let (dir_part, name_part) = match partial.rfind('/') { Some(i) => (&partial[..=i], &partial[i + 1..]), None => ("", partial) };
    if dir_part.split('/').any(|seg| seg == "..") { return Some(vec![]); }
    let dir = root.join(dir_part);
    let mut entries: Vec<(String, bool)> = std::fs::read_dir(&dir).ok()?.flatten().filter_map(|e| {
        let name = e.file_name().to_string_lossy().into_owned();
        let is_dir = e.file_type().ok()?.is_dir();
        (!name.starts_with('.') && name.to_lowercase().starts_with(&name_part.to_lowercase()) && (is_dir || kind == "file")).then_some((name, is_dir))
    }).collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.to_lowercase().cmp(&b.0.to_lowercase())));
    entries.truncate(30);
    Some(entries.into_iter().map(|(name, is_dir)| item(format!("@{kind}:{dir_part}{name}{}", if is_dir { "/" } else { "" }), if is_dir { "Folder" } else { "File" })).collect())
}

/// Keys the composer hands to the popup while it is open.
#[derive(Clone, Debug)]
pub(super) enum Action { Up, Down, Pick(usize), Accept, Dismiss }

pub(super) fn update(app: &mut Hermes, action: Action) -> Task<Message> {
    let list = items(app);
    if list.is_empty() { return Task::none(); }
    match action {
        Action::Up => app.slash.index = (app.slash.index + list.len() - 1) % list.len(),
        Action::Down => app.slash.index = (app.slash.index + 1) % list.len(),
        Action::Dismiss => app.slash.dismissed_for = Some(app.composer.text().trim_end().to_owned()),
        Action::Pick(i) => { app.slash.index = i; return update(app, Action::Accept); }
        Action::Accept => {
            let item = &list[app.slash.index.min(list.len() - 1)];
            if item.group == "References" {
                let text = app.composer.text();
                let line = text.trim_end_matches('\n');
                let token = reference_token(line).unwrap_or_default();
                let open = item.name.ends_with('/') || item.name.ends_with(':');
                let replaced = format!("{}{}{}", &line[..line.len() - token.len()], item.name, if open { "" } else { " " });
                app.composer = text_editor::Content::with_text(&replaced);
                app.composer.perform(text_editor::Action::Move(text_editor::Motion::DocumentEnd));
                app.slash.index = 0;
                return iced::widget::operation::focus(super::COMPOSER_ID);
            }
            // A complete command that takes no argument runs on Enter.
            let typed = app.composer.text().trim().to_lowercase();
            let takes_arg = COMMANDS.iter().find(|c| c.0 == item.name).is_none_or(|c| c.3);
            if typed == item.name && !takes_arg { return app.update(Message::Send); }
            app.composer = text_editor::Content::with_text(&format!("{} ", item.name));
            app.composer.perform(text_editor::Action::Move(text_editor::Motion::DocumentEnd));
            app.slash.index = 0;
            return iced::widget::operation::focus(super::COMPOSER_ID);
        }
    }
    Task::none()
}

/// Runs a "/command args" line. Returns None when the text is not a slash command.
pub(super) fn run(app: &mut Hermes, line: &str) -> Option<Task<Message>> {
    let line = line.trim();
    if !line.starts_with('/') || line.starts_with("//") { return None; }
    let (word, arg) = line.split_once(char::is_whitespace).map_or((line, ""), |(w, a)| (w, a.trim()));
    let word = word.to_lowercase();
    let canonical = COMMANDS.iter().find(|(name, aliases, _, _)| *name == word || aliases.contains(&word.as_str())).map(|c| c.0);
    let clear = |app: &mut Hermes| { app.composer = text_editor::Content::new(); let _ = app.save_draft(); app.slash = State::default(); };
    let task = match canonical {
        Some("/new") => { clear(app); app.update(Message::NewChat) }
        Some("/stop") => { clear(app); app.update(Message::Stop) }
        Some("/retry") => {
            let Some(last) = app.history.turns.last().map(|t| t.prompt.clone()) else { app.note("Nothing to retry yet."); return Some(Task::none()); };
            app.composer = text_editor::Content::with_text(&last); app.slash = State::default();
            app.update(Message::Send)
        }
        Some("/title") => {
            clear(app);
            if arg.is_empty() { app.update(Message::RenameStart) } else { app.rename = Some(arg.to_owned()); app.update(Message::RenameSave) }
        }
        Some("/model") => { clear(app); app.update(Message::Models(models::Action::Open)) }
        Some("/wake") => {
            clear(app);
            match arg.to_lowercase().as_str() {
                "on" => app.voice_update(super::voice::Action::Wake(true)),
                "off" => app.voice_update(super::voice::Action::Wake(false)),
                _ => { app.note(if app.voice.preferences.wake_word { if app.voice.wake_listening { "Wake word on · listening for “hey Hermes”" } else { "Wake word on · paused while you talk" } } else { "Wake word off · /wake on to listen for “hey Hermes”" }); Task::none() }
            }
        }
        Some("/reasoning") => {
            clear(app);
            let level = arg.to_lowercase();
            if level.is_empty() { app.note(format!("Reasoning effort: {}", reasoning::effective(app))); return Some(Task::none()); }
            if !["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"].contains(&level.as_str()) { app.error("Use /reasoning minimal|low|medium|high|xhigh|max|none."); return Some(Task::none()); }
            app.models.options.reasoning_effort = Some(level.clone());
            match app.models.save_options(&app.config) { Ok(()) => app.note(format!("Reasoning effort set to {level}.")), Err(e) => app.error(e) }
            Task::none()
        }
        Some("/approvals") | Some("/yolo") => {
            clear(app);
            let mode = if word == "/yolo" { if app.profile_settings.saved.approval_mode == "off" { "smart".to_owned() } else { "off".to_owned() } } else { arg.to_lowercase() };
            if mode.is_empty() { app.note(format!("Approval mode: {}", app.profile_settings.saved.approval_mode)); return Some(Task::none()); }
            let Some(value) = super::statusbar::APPROVAL_MODES.iter().find(|(v, _, _)| *v == mode).map(|(v, _, _)| *v) else { app.error("Use /approvals manual|smart|off."); return Some(Task::none()); };
            let task = app.update(Message::Statusbar(super::statusbar::Action::SetApproval(value)));
            app.note(format!("Approval mode: {value}")); task
        }
        Some("/personality") => {
            clear(app);
            let name = if arg.is_empty() { "default" } else { arg };
            if name != "default" && !hermes_engine::settings::PERSONALITIES.contains_key(name) {
                app.error(format!("Unknown personality. Try: {}", hermes_engine::settings::PERSONALITIES.keys().cloned().collect::<Vec<_>>().join(", "))); return Some(Task::none());
            }
            app.update(Message::ProfileSetting(super::profile_settings::Action::Set("personality", name.into(), settings::View::Chat)))
        }
        Some("/skin") => {
            clear(app);
            let skins = super::appearance::Skin::ALL;
            let current = skins.iter().position(|s| *s == app.appearance.preferences.skin).unwrap_or(0);
            let next = if arg.is_empty() { skins[(current + 1) % skins.len()] } else {
                match skins.iter().find(|s| s.label().eq_ignore_ascii_case(arg)) { Some(s) => *s, None => { app.error("Unknown theme."); return Some(Task::none()); } } };
            app.update(Message::Appearance(super::appearance::Action::Skin(next)))
        }
        Some("/queue") => {
            if arg.is_empty() { clear(app); app.note("Use /queue <prompt>."); return Some(Task::none()); }
            app.composer = text_editor::Content::with_text(arg); app.slash = State::default();
            app.update(Message::QueueCurrent)
        }
        Some("/steer") => {
            if arg.is_empty() { clear(app); app.note("Use /steer <message>."); return Some(Task::none()); }
            app.composer = text_editor::Content::with_text(arg); app.slash = State::default();
            app.update(Message::Send)
        }
        Some("/resume") => { clear(app); app.update(Message::OpenCommands) }
        Some("/palette") => { clear(app); app.update(Message::OpenCommands) }
        Some("/status") => {
            clear(app);
            let title = app.current.as_ref().map_or("New chat".to_owned(), |c| c.title.clone());
            app.note(format!("{title} · {} turns · model {} · approvals {}", app.history.turns.len(), app.models.model_label(&app.config), app.profile_settings.saved.approval_mode));
            Task::none()
        }
        Some("/config") => { clear(app); app.update(Message::SettingsView(settings::View::Models)) }
        Some("/agents") => { clear(app); app.update(Message::Navigate(Screen::Agents)) }
        Some("/cron") => { clear(app); app.update(Message::Navigate(Screen::Cron)) }
        Some("/skills") | Some("/tools") => { clear(app); app.update(Message::Navigate(Screen::Tools)) }
        Some("/plan") | Some("/init") | Some("/review") | Some("/btw") | Some("/learn") | Some("/refine") | Some("/goal") | Some("/subgoal") => {
            let instruction = match canonical.unwrap() {
                "/plan" => format!("Write a markdown implementation plan for the following and save it under .hermes/plans/. Do not execute anything else.\n\n{arg}"),
                "/init" => "Scan this project and generate or update AGENTS.md with concise project instructions for agents.".to_owned(),
                "/review" => format!("Independently review the work just discussed (code, docs or PR). Report concrete issues, most severe first.{}", if arg.is_empty() { String::new() } else { format!("\n\nFocus: {arg}") }),
                "/btw" => format!("Side question — answer it without changing the current task:\n\n{arg}"),
                "/learn" => format!("Learn a reusable skill from the following and write it as a SKILL.md I can install:\n\n{arg}"),
                "/refine" => "Review this conversation now and save the lessons worth keeping to memory.".to_owned(),
                "/goal" => format!("Standing goal for this session — keep working on it across turns until it is achieved:\n\n{arg}"),
                _ => format!("Add this criterion to the active goal:\n\n{arg}"),
            };
            if matches!(canonical, Some("/plan") | Some("/btw") | Some("/learn") | Some("/goal") | Some("/subgoal")) && arg.is_empty() {
                clear(app); app.note(format!("Use {} <text>.", canonical.unwrap())); return Some(Task::none());
            }
            app.composer = text_editor::Content::with_text(&instruction); app.slash = State::default();
            app.update(Message::Send)
        }
        Some("/diff") => { clear(app); app.update(Message::Navigate(Screen::Git)) }
        Some("/context") => {
            clear(app);
            if !app.statusbar.shows("context-usage") { let _ = app.update(Message::Statusbar(super::statusbar::Action::ToggleItem("context-usage"))); }
            app.note("Context usage is shown in the status bar."); Task::none()
        }
        Some("/debug") => {
            clear(app);
            let report = format!("Hermes Desktop {} (native Rust)\n{} / {}\nmodel {}\napprovals {}\nturns {}", env!("CARGO_PKG_VERSION"), std::env::consts::OS, std::env::consts::ARCH,
                app.config.model, app.profile_settings.saved.approval_mode, app.history.turns.len());
            app.note("Debug report copied. It has versions and settings only — no message text.");
            iced::clipboard::write(report)
        }
        Some("/export") | Some("/import") => { clear(app); app.update(Message::SettingsView(settings::View::Backup)) }
        Some("/handoff") => { clear(app); app.update(Message::Navigate(Screen::Messaging)) }
        Some("/journey") | Some("/memory") => { clear(app); app.update(Message::SettingsView(settings::View::Memory)) }
        Some("/rollback") => { clear(app); app.update(Message::SettingsView(settings::View::Safety)) }
        Some("/subscription") | Some("/topup") | Some("/usage") => { clear(app); app.update(Message::SettingsView(settings::View::Billing)) }
        Some("/profile") => { clear(app); app.update(Message::Bot(super::bots::Action::Roster(true))) }
        Some("/pause") => { clear(app); app.queue.parked = true; let _ = app.save_queue(); let t = app.update(Message::Stop); app.note("Paused. Queued messages wait until you resume."); t }
        Some("/version") => { clear(app); app.note(format!("Hermes Desktop {} · native Rust build", env!("CARGO_PKG_VERSION"))); Task::none() }
        Some("/whoami") => { clear(app); app.note("You are the local owner of this desktop: all commands are available."); Task::none() }
        Some("/save") => {
            clear(app);
            let title = app.current.as_ref().map_or("chat".to_owned(), |c| c.title.clone());
            let safe: String = title.chars().map(|c| if c.is_alphanumeric() { c } else { '-' }).collect();
            let dir = std::env::var_os("HOME").map(|h| std::path::Path::new(&h).join("Downloads")).unwrap_or_else(std::env::temp_dir);
            let path = dir.join(format!("hermes-{safe}.json"));
            match serde_json::to_vec_pretty(&app.history).map_err(|e| e.to_string()).and_then(|b| std::fs::write(&path, b).map_err(|e| e.to_string())) {
                Ok(()) => app.note(format!("Saved the transcript to {}", path.display())), Err(e) => app.error(format!("Could not save: {e}")),
            }
            Task::none()
        }
        Some(name @ ("/bg" | "/loop" | "/heartbeat")) => { clear(app); app.note(format!("{name} runs recurring or background prompts in the Hermes backend; use Scheduled jobs here.")); app.update(Message::Navigate(Screen::Cron)) }
        Some(name) if !matches!(name, "/help") => {
            clear(app);
            app.note(format!("{name} needs a part of the Hermes backend this native build does not run yet."));
            Task::none()
        }
        Some("/help") => {
            app.composer = text_editor::Content::with_text("/"); app.slash = State::default();
            app.note("Slash commands: type / to browse them.");
            iced::widget::operation::focus(super::COMPOSER_ID)
        }
        _ => {
            // "/<skill> …": send the request with that skill loaded.
            let slug = &word[1..];
            let installed = app.library.as_ref().and_then(|l| hermes_engine::skill_catalog::status(l.root()).ok())
                .and_then(|s| s.installed.iter().find(|id| id.rsplit('/').next() == Some(slug) && !s.disabled.contains(*id)).cloned());
            if TERMINAL_ONLY.contains(&word.as_str()) { app.note(format!("{word} is a Hermes terminal (CLI) command.")); return Some(Task::none()); }
            match installed {
                Some(id) => {
                    let request = if arg.is_empty() { "Follow this skill's instructions.".to_owned() } else { arg.to_owned() };
                    app.composer = text_editor::Content::with_text(&format!("Use the \"{slug}\" skill (call skill_view with name \"catalog:{id}\" first), then: {request}"));
                    app.slash = State::default();
                    app.update(Message::Send)
                }
                None => { app.error(format!("Unknown command {word}. Type / to see what's available.")); Task::none() }
            }
        }
    };
    Some(task)
}

/// The popup above the composer (ComposerTriggerPopover): grouped rows,
/// the command in medium weight, its description muted, the active row tinted.
pub(super) fn popup(app: &Hermes) -> Option<Element<'_, Message>> {
    let list = items(app);
    if list.is_empty() { return None; }
    let p = ui::palette(app.dark);
    let active = app.slash.index.min(list.len() - 1);
    let mut rows = column![].spacing(1);
    let mut last_group = "";
    for (index, item) in list.into_iter().enumerate() {
        if item.group != last_group {
            rows = rows.push(container(text(item.group.to_uppercase()).size(10).font(ui::medium()).color(Color { a: 0.55, ..p.foreground })).padding([6, 8]));
            last_group = item.group;
        }
        let selected = index == active;
        rows = rows.push(button(row![text(item.name.clone()).size(13).font(ui::medium()), Space::new().width(12),
                text(item.description.clone()).size(12).color(p.muted), Space::new().width(Fill)].align_y(Alignment::Center))
            .padding([5, 8]).width(Fill).on_press(Message::Slash(Action::Pick(index)))
            .style(move |_, status| button::Style {
                background: (selected || status == button::Status::Hovered).then(|| Color { a: if selected { 0.08 } else { 0.05 }, ..p.foreground }.into()),
                text_color: p.foreground, border: iced::Border { radius: 4.0.into(), ..Default::default() }, ..Default::default() }));
    }
    Some(container(scrollable(rows).height(iced::Length::Shrink)).max_height(280).padding(4).width(Fill)
        .style(move |_| ui::panel(p.card, 6.0, Some(Color { a: 0.1, ..p.foreground }))).into())
}
