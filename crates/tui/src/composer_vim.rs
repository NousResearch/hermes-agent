//! Opt-in composer vim. Off by default: Esc stays interrupt/quit.
//! `/vim` enters Normal. Esc in Insert → Normal. Esc in Normal → leave vim.
//! Pending keys: `dd`, `dw`, `gg`. `G` jumps to the bottom.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui_textarea::{CursorMove, TextArea};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VimMode {
    Normal,
    Insert,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VimState {
    pub mode: VimMode,
    pub pending: Option<char>,
}

impl VimState {
    pub fn normal() -> Self {
        Self {
            mode: VimMode::Normal,
            pending: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VimAction {
    Stay,
    Leave,
}

pub fn handle(vim: &mut VimState, key: KeyEvent, textarea: &mut TextArea<'_>) -> VimAction {
    if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('c')) {
        return VimAction::Stay;
    }
    match vim.mode {
        VimMode::Insert => match key.code {
            KeyCode::Esc => {
                vim.mode = VimMode::Normal;
                vim.pending = None;
                VimAction::Stay
            }
            KeyCode::Enter => {
                textarea.insert_newline();
                VimAction::Stay
            }
            _ => {
                textarea.input(key);
                VimAction::Stay
            }
        },
        VimMode::Normal => {
            if let Some(pend) = vim.pending.take() {
                return finish_pending(pend, key, textarea, vim);
            }
            match (key.modifiers, key.code) {
                (KeyModifiers::NONE, KeyCode::Esc) => VimAction::Leave,
                (KeyModifiers::NONE, KeyCode::Char('i')) => {
                    vim.mode = VimMode::Insert;
                    VimAction::Stay
                }
                (KeyModifiers::NONE, KeyCode::Char('a')) => {
                    textarea.move_cursor(CursorMove::Forward);
                    vim.mode = VimMode::Insert;
                    VimAction::Stay
                }
                (KeyModifiers::NONE, KeyCode::Char('I')) => {
                    textarea.move_cursor(CursorMove::Head);
                    vim.mode = VimMode::Insert;
                    VimAction::Stay
                }
                (KeyModifiers::NONE, KeyCode::Char('A')) => {
                    textarea.move_cursor(CursorMove::End);
                    vim.mode = VimMode::Insert;
                    VimAction::Stay
                }
                (KeyModifiers::NONE, KeyCode::Char('o')) => {
                    textarea.move_cursor(CursorMove::End);
                    textarea.insert_newline();
                    vim.mode = VimMode::Insert;
                    VimAction::Stay
                }
                (KeyModifiers::NONE, KeyCode::Char('h') | KeyCode::Left) => {
                    textarea.move_cursor(CursorMove::Back);
                    VimAction::Stay
                }
                (KeyModifiers::NONE, KeyCode::Char('j') | KeyCode::Down) => {
                    textarea.move_cursor(CursorMove::Down);
                    VimAction::Stay
                }
                (KeyModifiers::NONE, KeyCode::Char('k') | KeyCode::Up) => {
                    textarea.move_cursor(CursorMove::Up);
                    VimAction::Stay
                }
                (KeyModifiers::NONE, KeyCode::Char('l') | KeyCode::Right) => {
                    textarea.move_cursor(CursorMove::Forward);
                    VimAction::Stay
                }
                (KeyModifiers::NONE, KeyCode::Char('w')) => {
                    textarea.move_cursor(CursorMove::WordForward);
                    VimAction::Stay
                }
                (KeyModifiers::NONE, KeyCode::Char('b')) => {
                    textarea.move_cursor(CursorMove::WordBack);
                    VimAction::Stay
                }
                (KeyModifiers::NONE, KeyCode::Char('0') | KeyCode::Char('^')) => {
                    textarea.move_cursor(CursorMove::Head);
                    VimAction::Stay
                }
                (KeyModifiers::NONE, KeyCode::Char('$')) => {
                    textarea.move_cursor(CursorMove::End);
                    VimAction::Stay
                }
                (KeyModifiers::NONE, KeyCode::Char('x') | KeyCode::Delete) => {
                    textarea.delete_next_char();
                    VimAction::Stay
                }
                (KeyModifiers::NONE, KeyCode::Char('u')) => {
                    textarea.undo();
                    VimAction::Stay
                }
                (KeyModifiers::CONTROL, KeyCode::Char('r')) => {
                    textarea.redo();
                    VimAction::Stay
                }
                (KeyModifiers::NONE, KeyCode::Char('G')) => {
                    textarea.move_cursor(CursorMove::Bottom);
                    VimAction::Stay
                }
                (KeyModifiers::NONE, KeyCode::Char('d' | 'g')) => {
                    if let KeyCode::Char(c) = key.code {
                        vim.pending = Some(c);
                    }
                    VimAction::Stay
                }
                _ => VimAction::Stay,
            }
        }
    }
}

fn finish_pending(
    pend: char,
    key: KeyEvent,
    textarea: &mut TextArea<'_>,
    vim: &mut VimState,
) -> VimAction {
    vim.pending = None;
    match (pend, key.modifiers, key.code) {
        ('d', KeyModifiers::NONE, KeyCode::Char('d')) => {
            textarea.delete_line_by_end();
            textarea.move_cursor(CursorMove::Head);
            textarea.delete_line_by_head();
            VimAction::Stay
        }
        ('d', KeyModifiers::NONE, KeyCode::Char('w')) => {
            textarea.delete_next_word();
            VimAction::Stay
        }
        ('g', KeyModifiers::NONE, KeyCode::Char('g')) => {
            textarea.move_cursor(CursorMove::Top);
            textarea.move_cursor(CursorMove::Head);
            VimAction::Stay
        }
        (_, KeyModifiers::NONE, KeyCode::Esc) => VimAction::Stay,
        _ => VimAction::Stay,
    }
}

pub fn label(vim: VimState) -> &'static str {
    match vim.mode {
        VimMode::Normal => "vim",
        VimMode::Insert => "vim i",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_esc_goes_normal_and_normal_esc_leaves() {
        let mut ta = TextArea::from(["hello"]);
        let mut vim = VimState {
            mode: VimMode::Insert,
            pending: None,
        };
        assert_eq!(
            handle(&mut vim, KeyEvent::from(KeyCode::Esc), &mut ta),
            VimAction::Stay
        );
        assert_eq!(vim.mode, VimMode::Normal);
        assert_eq!(
            handle(&mut vim, KeyEvent::from(KeyCode::Esc), &mut ta),
            VimAction::Leave
        );
    }

    #[test]
    fn dd_deletes_line_gg_goes_top() {
        let mut ta = TextArea::from(["one", "two", "three"]);
        ta.move_cursor(CursorMove::Down);
        let mut vim = VimState::normal();
        let d = KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE);
        assert_eq!(handle(&mut vim, d, &mut ta), VimAction::Stay);
        assert_eq!(vim.pending, Some('d'));
        assert_eq!(handle(&mut vim, d, &mut ta), VimAction::Stay);
        assert!(vim.pending.is_none());
        assert_eq!(ta.lines().len(), 2);
        let g = KeyEvent::new(KeyCode::Char('g'), KeyModifiers::NONE);
        handle(&mut vim, g, &mut ta);
        handle(&mut vim, g, &mut ta);
        assert_eq!(ta.cursor(), (0, 0));
    }

    #[test]
    fn normal_i_enters_insert() {
        let mut ta = TextArea::from(["x"]);
        let mut vim = VimState::normal();
        let key = KeyEvent::new(KeyCode::Char('i'), KeyModifiers::NONE);
        assert_eq!(handle(&mut vim, key, &mut ta), VimAction::Stay);
        assert_eq!(vim.mode, VimMode::Insert);
    }
}
