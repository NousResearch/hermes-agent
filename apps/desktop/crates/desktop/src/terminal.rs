//! Native counterpart of src/app/right-sidebar/terminal. Each local tab owns a
//! genuine pseudoterminal; hiding the rail does not stop its shell.
use crate::appearance::text;
use iced::{
    Element, Fill, Size, Task,
    keyboard::{Key, key::Named},
    widget::{self, button, column, container, row, sensor, text_editor},
};
use portable_pty::{ChildKiller, CommandBuilder, MasterPty, PtySize, native_pty_system};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

const INITIAL_ROWS: u16 = 24;
const INITIAL_COLS: u16 = 80;
const EDITOR_ID: &str = "hermes-terminal-surface";

#[derive(Clone)]
pub enum Action {
    New,
    AddToChat(String),
    Select(u64),
    Close(u64),
    Input(Vec<u8>),
    Paste,
    Pasted(Option<String>),
    Display(text_editor::Action),
    Output(u64, Arc<[u8]>),
    End(u64, String),
    Resize(Size),
}
impl std::fmt::Debug for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Terminal action")
    }
}

struct Pty {
    master: Box<dyn MasterPty + Send>,
    input: std::sync::mpsc::SyncSender<Vec<u8>>,
    killer: Box<dyn ChildKiller + Send + Sync>,
}
impl Drop for Pty {
    fn drop(&mut self) {
        let _ = self.killer.kill();
    }
}

fn spawn_shell(
    command: CommandBuilder,
    rows: u16,
    cols: u16,
) -> Result<
    (
        Pty,
        Box<dyn Read + Send>,
        Box<dyn portable_pty::Child + Send + Sync>,
    ),
    String,
> {
    let pair = native_pty_system()
        .openpty(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|e| e.to_string())?;
    let reader = pair.master.try_clone_reader().map_err(|e| e.to_string())?;
    let mut writer = pair.master.take_writer().map_err(|e| e.to_string())?;
    let child = pair
        .slave
        .spawn_command(command)
        .map_err(|e| e.to_string())?;
    let killer = child.clone_killer();
    let (input, queue) = std::sync::mpsc::sync_channel::<Vec<u8>>(64);
    // A shell that stops reading must never block the desktop event loop.
    std::thread::spawn(move || {
        while let Ok(bytes) = queue.recv() {
            if writer
                .write_all(&bytes)
                .and_then(|_| writer.flush())
                .is_err()
            {
                break;
            }
        }
    });
    Ok((
        Pty {
            master: pair.master,
            input,
            killer,
        },
        reader,
        child,
    ))
}

struct Tab {
    workspace: Option<PathBuf>,
    name: String,
    pty: Option<Pty>,
    parser: vt100::Parser,
    editor: text_editor::Content,
    ended: Option<String>,
}
impl Tab {
    fn refresh(&mut self) {
        let (r, c) = self.parser.screen().cursor_position();
        let mut contents = self.parser.screen().contents();
        let n = contents.split('\n').count();
        if n <= r as usize {
            contents.extend(std::iter::repeat_n(
                '\n',
                (r as usize + 1).saturating_sub(n),
            ));
        }
        self.editor = text_editor::Content::with_text(&contents);
        self.editor.move_to(text_editor::Cursor {
            position: text_editor::Position {
                line: r.into(),
                column: c.into(),
            },
            selection: None,
        });
    }
}

pub struct TerminalPane {
    workspace: Option<PathBuf>,
    current: Option<u64>,
    tabs: BTreeMap<u64, Tab>,
    next: u64,
    error: Option<String>,
    rows: u16,
    cols: u16,
}
impl TerminalPane {
    pub fn new() -> Self {
        Self {
            workspace: None,
            current: None,
            tabs: BTreeMap::new(),
            next: 0,
            error: None,
            rows: INITIAL_ROWS,
            cols: INITIAL_COLS,
        }
    }
    pub fn set_workspace(&mut self, workspace: Option<PathBuf>) {
        if self.workspace == workspace {
            return;
        }
        self.workspace = workspace;
        self.current = self
            .tabs
            .iter()
            .rev()
            .find(|(_, t)| t.workspace == self.workspace)
            .map(|(id, _)| *id);
        self.error = None;
    }
    pub fn ensure_shell(&mut self) -> Task<Action> {
        if self.current.is_none() {
            self.update(Action::New)
        } else {
            widget::operation::focus(EDITOR_ID)
        }
    }
    fn current_mut(&mut self) -> Option<&mut Tab> {
        self.tabs.get_mut(&self.current?)
    }

    fn new_shell(&mut self) -> Result<Task<Action>, String> {
        if self.tabs.len() >= 16 {
            return Err("Close a terminal before opening another.".into());
        }
        let program = std::env::var_os(if cfg!(windows) { "COMSPEC" } else { "SHELL" })
            .unwrap_or_else(|| {
                if cfg!(windows) {
                    "cmd.exe".into()
                } else {
                    "/bin/sh".into()
                }
            });
        let name = Path::new(&program)
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or("Terminal")
            .to_owned();
        let mut cmd = CommandBuilder::new(&program);
        if let Some(folder) = &self.workspace {
            cmd.cwd(folder);
        } else if let Some(dir) = std::env::home_dir() {
            cmd.cwd(dir);
        }
        cmd.env("TERM", "xterm-256color");
        cmd.env_remove("HERMES_RUST_API_KEY");
        cmd.env_remove("HERMES_RUST_MCP_TOKEN");
        let (pty, mut output, mut child) = spawn_shell(cmd, self.rows, self.cols)?;
        self.next += 1;
        let id = self.next;
        self.tabs.insert(
            id,
            Tab {
                workspace: self.workspace.clone(),
                name,
                pty: Some(pty),
                parser: vt100::Parser::new(self.rows, self.cols, 1000),
                editor: text_editor::Content::new(),
                ended: None,
            },
        );
        self.current = Some(id);
        let (sender, rx) = async_channel::bounded(24);
        // Drain final PTY bytes before reporting exit so the final prompt or
        // command result cannot be overtaken by a separate wait task.
        std::thread::spawn(move || {
            let mut buf = [0_u8; 8192];
            loop {
                match output.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        if sender
                            .send_blocking(Action::Output(id, Arc::from(&buf[..n])))
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                }
            }
            let status = child
                .wait()
                .map(|s| s.exit_code().to_string())
                .unwrap_or_else(|_| "unknown".into());
            let _ = sender.send_blocking(Action::End(id, status));
        });
        Ok(Task::batch([
            Task::run(rx, |event| event),
            widget::operation::focus(EDITOR_ID),
        ]))
    }

    pub fn update(&mut self, action: Action) -> Task<Action> {
        match action {
            Action::New => match self.new_shell() {
                Ok(task) => return task,
                Err(e) => self.error = Some(e),
            },
            Action::Select(id)
                if self
                    .tabs
                    .get(&id)
                    .is_some_and(|t| t.workspace == self.workspace) =>
            {
                self.current = Some(id);
                return widget::operation::focus(EDITOR_ID);
            }
            Action::Select(_) => {}
            Action::Close(id) => {
                self.tabs.remove(&id);
                if self.current == Some(id) {
                    self.current = self
                        .tabs
                        .iter()
                        .rev()
                        .find(|(_, t)| t.workspace == self.workspace)
                        .map(|(id, _)| *id);
                }
            }
            Action::Output(id, bytes) => {
                if let Some(tab) = self.tabs.get_mut(&id) {
                    tab.parser.process(&bytes);
                    tab.refresh();
                }
            }
            Action::End(id, code) => {
                if let Some(tab) = self.tabs.get_mut(&id) {
                    tab.pty = None;
                    tab.ended = Some(format!("Process exited with code {code}"));
                }
            }
            Action::AddToChat(_) => {}
            Action::Display(text_editor::Action::Scroll { lines }) => {
                if let Some(tab) = self.current_mut() {
                    let offset = tab
                        .parser
                        .screen()
                        .scrollback()
                        .saturating_add_signed(-(lines as isize));
                    tab.parser.screen_mut().set_scrollback(offset);
                    tab.refresh();
                }
            }
            Action::Display(action) => {
                if let Some(tab) = self.current_mut() {
                    if !matches!(action, text_editor::Action::Edit(_)) {
                        tab.editor.perform(action);
                    }
                }
            }
            Action::Input(bytes) => {
                if bytes.len() > 1_048_576 {
                    self.error = Some("Terminal input exceeds 1 MiB".into());
                } else if let Some(tab) = self.current_mut() {
                    if tab.parser.screen().scrollback() > 0 {
                        tab.parser.screen_mut().set_scrollback(0);
                        tab.refresh();
                    }
                    if let Some(pty) = &tab.pty
                        && let Err(error) = pty.input.try_send(bytes)
                    {
                        self.error = Some(format!("Terminal input unavailable: {error}"));
                    }
                }
            }
            Action::Paste => return iced::clipboard::read().map(Action::Pasted),
            Action::Pasted(Some(data)) => {
                let bytes = if self
                    .current
                    .and_then(|id| self.tabs.get(&id))
                    .is_some_and(|t| t.parser.screen().bracketed_paste())
                {
                    format!("\x1b[200~{data}\x1b[201~").into_bytes()
                } else {
                    data.into_bytes()
                };
                return self.update(Action::Input(bytes));
            }
            Action::Pasted(None) => {}
            Action::Resize(size) => {
                let cols = ((size.width - 12.0) / 7.8).floor().clamp(12.0, 300.0) as u16;
                let rows = ((size.height - 10.0) / 17.0).floor().clamp(3.0, 150.0) as u16;
                self.cols = cols;
                self.rows = rows;
                if let Some(tab) = self.current_mut() {
                    if tab.parser.screen().size() != (rows, cols) {
                        if let Some(pty) = &tab.pty {
                            let _ = pty.master.resize(PtySize {
                                rows,
                                cols,
                                pixel_width: 0,
                                pixel_height: 0,
                            });
                        }
                        tab.parser.screen_mut().set_size(rows, cols);
                        tab.refresh();
                    }
                }
            }
        }
        Task::none()
    }

    pub fn view(&self, shortcuts: &crate::shortcuts::Bindings, capturing: bool) -> Element<'_, Action> {
        let shortcuts = shortcuts.clone();
        let mut rail = row![].spacing(2);
        for (&id, t) in self
            .tabs
            .iter()
            .filter(|(_, tab)| tab.workspace == self.workspace)
        {
            rail = rail
                .push(
                    button(text(&t.name).size(11))
                        .on_press(Action::Select(id))
                        .style(button::text),
                )
                .push(
                    button(text("×").size(12))
                        .on_press(Action::Close(id))
                        .style(button::text),
                );
        }
        rail = rail.push(
            button(text("+").size(15))
                .on_press(Action::New)
                .style(button::text),
        );
        let mut body = column![rail].spacing(3);
        if let Some(error) = &self.error {
            body = body.push(text(error).size(12));
        }
        if let Some(tab) = self.current.and_then(|id| self.tabs.get(&id)) {
            if let Some(selection) = tab.editor.selection()
                && !selection.trim().is_empty()
            {
                body = body.push(
                    button(text("Add to chat").size(11))
                        .on_press(Action::AddToChat(selection))
                        .style(button::text),
                );
            }
            let editor = text_editor(&tab.editor)
                .id(EDITOR_ID)
                .on_action(Action::Display)
                .key_binding(move |press| {
                    if capturing || shortcuts.action_for_key(&press.key, press.physical_key, press.modifiers, true).is_some() { None }
                    else { terminal_key(press) }
                })
                .font(crate::appearance::terminal_font())
                .size(13)
                .line_height(iced::widget::text::LineHeight::Absolute(17.0.into()))
                .padding(5)
                .height(Fill)
                .style(|theme, status| {
                    let mut s = text_editor::default(theme, status);
                    s.border = iced::Border::default();
                    s
                });
            body = body.push(
                sensor(editor)
                    .on_resize(Action::Resize)
                    .on_show(Action::Resize),
            );
            if let Some(end) = &tab.ended {
                body = body.push(text(end).size(11));
            }
        } else {
            body = body.push(container(button("New terminal").on_press(Action::New)).center(Fill));
        }
        container(body).padding(4).width(Fill).height(Fill).into()
    }
}

fn terminal_key(key: text_editor::KeyPress) -> Option<text_editor::Binding<Action>> {
    use text_editor::Binding;
    if key.modifiers.logo() || (key.modifiers.control() && key.modifiers.shift()) {
        return match key.key.as_ref() {
            Key::Character("c") => Some(Binding::Copy),
            Key::Character("v") => Some(Binding::Custom(Action::Paste)),
            Key::Character("a") => Some(Binding::SelectAll),
            _ => None,
        };
    }
    if key.modifiers.control() {
        if let Key::Character(ch) = key.key.as_ref() {
            if ch.len() == 1 && ch.as_bytes()[0].is_ascii_alphabetic() {
                return Some(Binding::Custom(Action::Input(vec![
                    ch.as_bytes()[0].to_ascii_uppercase() & 31,
                ])));
            }
        }
    }
    let bytes = match key.key.as_ref() {
        Key::Named(Named::Enter) => b"\r".to_vec(),
        Key::Named(Named::Backspace) => vec![127],
        Key::Named(Named::Tab) => b"\t".to_vec(),
        Key::Named(Named::Escape) => vec![27],
        Key::Named(Named::ArrowUp) => b"\x1b[A".to_vec(),
        Key::Named(Named::ArrowDown) => b"\x1b[B".to_vec(),
        Key::Named(Named::ArrowRight) => b"\x1b[C".to_vec(),
        Key::Named(Named::ArrowLeft) => b"\x1b[D".to_vec(),
        Key::Named(Named::Home) => b"\x1b[H".to_vec(),
        Key::Named(Named::End) => b"\x1b[F".to_vec(),
        Key::Named(Named::Delete) => b"\x1b[3~".to_vec(),
        _ => key.text?.as_bytes().to_vec(),
    };
    Some(Binding::Custom(Action::Input(bytes)))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    #[test]
    fn real_pty_inputs_and_resizes_are_seen_by_the_child() {
        let mut cmd = CommandBuilder::new("/bin/sh");
        cmd.args([
            "-c",
            "read first; stty size; printf 'answer=%s\\n' \"$first\"",
        ]);
        let (pty, mut out, mut child) = spawn_shell(cmd, 24, 80).unwrap();
        pty.master
            .resize(PtySize {
                rows: 16,
                cols: 92,
                pixel_width: 0,
                pixel_height: 0,
            })
            .unwrap();
        pty.input.send(b"native-pty\n".to_vec()).unwrap();
        let mut result = Vec::new();
        let _ = out.read_to_end(&mut result);
        assert!(child.wait().unwrap().success());
        let text = String::from_utf8_lossy(&result);
        assert!(
            text.contains("16 92") && text.contains("answer=native-pty"),
            "{text}"
        );
    }
    #[test]
    fn emulator_applies_terminal_control_sequences() {
        let mut p = vt100::Parser::new(4, 20, 0);
        p.process(b"old\r\x1b[2Knew");
        assert_eq!(p.screen().contents(), "new");
    }
}
