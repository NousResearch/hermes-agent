//! Files the user chose in the composer. Drafts belong to a single native chat.
use hermes_engine::Attachment;
use iced::widget::{button, column, row, text};
use iced::{Alignment, Element};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};
use tempfile::NamedTempFile;

#[derive(Clone)]
pub(super) enum Action {
    Choose,
    Selected(String, Result<Option<PathBuf>, String>),
    Drop(PathBuf),
    Captured(String, Result<Attachment, String>),
    Remove(usize),
}
impl std::fmt::Debug for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Attachment action")
    }
}

#[derive(Default)]
pub(super) struct DraftFiles {
    pub(super) files: Vec<Attachment>,
    pub(super) pending: usize,
}

impl DraftFiles {
    pub(super) fn load(directory: &Path) -> Result<Self, String> {
        let bytes = match fs::read(directory.join("draft-attachments.json")) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(e.to_string()),
        };
        if bytes.len() > 128 * 1024 {
            return Err("Invalid attachment draft".into());
        }
        let files: Vec<Attachment> = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        Attachment::validate_set(&files).map_err(|e| e.to_string())?;
        Ok(Self { files, pending: 0 })
    }
    pub(super) fn replace(&mut self, dir: &Path, files: Vec<Attachment>) -> Result<(), String> {
        Attachment::validate_set(&files).map_err(|e| e.to_string())?;
        let mut temp = NamedTempFile::new_in(dir).map_err(|e| e.to_string())?;
        serde_json::to_writer(&mut temp, &files).map_err(|e| e.to_string())?;
        temp.flush()
            .and_then(|_| temp.as_file().sync_all())
            .map_err(|e| e.to_string())?;
        temp.persist(dir.join("draft-attachments.json"))
            .map_err(|e| e.to_string())?;
        self.files = files;
        Ok(())
    }
    pub(super) fn view(&self) -> Element<'_, Action> {
        let mut list = column![].spacing(2);
        for (index, file) in self.files.iter().enumerate() {
            list = list.push(
                row![
                    text(&file.name).size(12),
                    button(text("×").size(13))
                        .padding([1, 5])
                        .on_press(Action::Remove(index))
                        .style(button::text)
                ]
                .spacing(4)
                .align_y(Alignment::Center),
            );
        }
        if self.pending > 0 {
            list = list.push(text("Adding attachment…").size(11));
        }
        list.into()
    }
}

pub(super) fn choose() -> Result<Option<PathBuf>, String> {
    crate::picker::choose("Attach a file",false)
}

impl super::Hermes {
    pub(super) fn attachment_action(&mut self, action: Action) -> iced::Task<super::Message> {
        use super::Message;
        use iced::Task;
        let Some(chat) = self.current.as_ref() else {
            return Task::none();
        };
        let id = chat.id.clone();
        let directory = chat.directory.clone();
        match action {
            Action::Choose => {
                self.attach_menu = false;
                return Task::perform(
                    async {
                        tokio::task::spawn_blocking(choose)
                            .await
                            .map_err(|e| e.to_string())
                            .and_then(|r| r)
                    },
                    move |result| Message::Attachment(Action::Selected(id.clone(), result)),
                );
            }
            Action::Selected(owner, result) => {
                if owner != id {
                    return Task::none();
                }
                match result {
                    Ok(Some(path)) => return self.attachment_action(Action::Drop(path)),
                    Err(e) => self.error(e),
                    Ok(None) => {}
                }
            }
            Action::Drop(path) => {
                if self.screen != super::Screen::Chat {
                    return Task::none();
                }
                if self.draft_files.files.len() + self.draft_files.pending >= 8 {
                    self.error("Attach up to eight files per message");
                    return Task::none();
                }
                self.draft_files.pending += 1;
                return Task::perform(
                    async move {
                        tokio::task::spawn_blocking(move || {
                            Attachment::capture(&directory, &path).map_err(|e| format!("{e:#}"))
                        })
                        .await
                        .map_err(|e| e.to_string())
                        .and_then(|r| r)
                    },
                    move |result| Message::Attachment(Action::Captured(id.clone(), result)),
                );
            }
            Action::Captured(owner, result) => {
                if owner != id {
                    return Task::none();
                }
                self.draft_files.pending = self.draft_files.pending.saturating_sub(1);
                match result {
                    Ok(file) => {
                        let mut files = self.draft_files.files.clone();
                        if !files.contains(&file) {
                            files.push(file);
                            if let Err(e) = self.draft_files.replace(&directory, files) {
                                self.error(e);
                            }
                        }
                    }
                    Err(e) => self.error(e),
                }
            }
            Action::Remove(index) => {
                if index < self.draft_files.files.len() {
                    let mut files = self.draft_files.files.clone();
                    files.remove(index);
                    if let Err(e) = self.draft_files.replace(&directory, files) {
                        self.error(e);
                    }
                }
            }
        }
        Task::none()
    }
}
pub(super) fn drop_event(
    event: iced::Event,
    _status: iced::event::Status,
    _window: iced::window::Id,
) -> Option<super::Message> {
    if let iced::Event::Window(iced::window::Event::FileDropped(path)) = event {
        Some(super::Message::Attachment(Action::Drop(path)))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn draft_survives_reopen_and_removal_is_durable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("attached.txt");
        fs::write(&path, "synthetic note").unwrap();
        let file = Attachment::capture(dir.path(), &path).unwrap();
        let mut draft = DraftFiles::load(dir.path()).unwrap();
        draft.replace(dir.path(), vec![file.clone()]).unwrap();
        assert_eq!(DraftFiles::load(dir.path()).unwrap().files, vec![file]);
        draft.replace(dir.path(), vec![]).unwrap();
        assert!(DraftFiles::load(dir.path()).unwrap().files.is_empty());
    }
}
