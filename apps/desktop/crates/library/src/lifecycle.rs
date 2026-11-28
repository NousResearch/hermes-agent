//! Durable native session lifecycle derived from official v0.23.3
//! (caafe5389) sidebar archive/pin flows. Archive is a metadata operation;
//! conversation bytes and manual archive state survive subsequent opens.
use super::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArchiveFilter {
    #[default]
    Active,
    Archived,
    All,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SessionQuery {
    pub project_id: Option<String>,
    pub archive: ArchiveFilter,
    pub pinned: Option<bool>,
    /// Case-insensitive title, native ID, or imported Hermes origin ID.
    pub search: String,
    /// Leave out native chats with no messages, as the original sidebar does.
    pub hide_empty: bool,
}

impl Library {
    /// Archive by last activity, excluding the foreground conversation and pins.
    pub fn archive_inactive(&self,days:u32,current:Option<&str>)->Result<()>{
        ensure!(days>0 && days<=36500,"Invalid inactive session age");
        let cutoff=Utc::now()-chrono::Duration::days(i64::from(days));
        self.with_catalog(true,|catalog|{
            for record in &mut catalog.sessions {
                if record.archived || record.pinned || current==Some(&record.id) || catalog.active_session.as_ref()==Some(&record.id){continue}
                if chrono::DateTime::parse_from_rfc3339(&record.updated_at).is_ok_and(|stamp|stamp<cutoff){
                    let lock=OpenOptions::new().read(true).write(true).create(true).truncate(false).open(self.checked_directory(&record.id)?.join("session.lock"))?;
                    if lock.try_lock_exclusive().is_ok(){record.archived=true;}
                }
            }Ok(())
        })
    }

    /// Journal removal before unlinking conversation files. User-created files
    /// in a chat's default workspace are retained outside the session directory.
    pub fn delete_archived(&self,id:&str)->Result<()>{
        let directory=self.session_dir(id)?;
        let lock=OpenOptions::new().read(true).write(true).create(true).truncate(false).open(directory.join("session.lock"))?;
        lock.try_lock_exclusive().context("Close the conversation before deleting it")?;
        self.with_catalog(true,|catalog|{
            let record=find_session(catalog,id)?;
            ensure!(record.archived && catalog.active_session.as_deref()!=Some(id),"Only archived, inactive conversations can be deleted here");
            catalog.sessions.retain(|s|s.id!=id);
            catalog.pinned_order.retain(|p|p!=id);
            catalog.pending_deletion.push(id.to_owned());
            Ok(())
        })?;
        self.finish_deletions()
    }
    pub(crate) fn finish_deletions(&self)->Result<()> {
        if self.with_catalog(false,|c|Ok(c.pending_deletion.is_empty()))? {return Ok(());}
        self.with_catalog(true,|catalog|{
            for id in &catalog.pending_deletion {
                ensure!(safe_id(id) && !catalog.sessions.iter().any(|r|&r.id==id),"Invalid pending session removal");
                let dir=self.root.join("sessions").join(id);
                match fs::symlink_metadata(&dir){
                    Ok(meta)=>ensure!(meta.file_type().is_dir(),"Invalid archived session directory"),
                    Err(e) if e.kind()==std::io::ErrorKind::NotFound=>continue,
                    Err(e)=>return Err(e.into()),
                }
                let work=dir.join("workspace");
                let preserve=match fs::symlink_metadata(&work){
                    Ok(meta)=>meta.file_type().is_symlink() || !meta.file_type().is_dir() || work.read_dir()?.next().transpose()?.is_some(),
                    Err(e) if e.kind()==std::io::ErrorKind::NotFound=>false,
                    Err(e)=>return Err(e.into()),
                };
                if preserve {
                    let keep=self.root.join("retained-workspaces");
                    fs::create_dir_all(&keep)?;ensure!(fs::symlink_metadata(&keep)?.file_type().is_dir(),"Invalid retained workspaces directory");
                    fs::rename(&work,keep.join(id)).context("Could not preserve project files")?;
                }
                fs::remove_dir_all(dir)?;
            }
            catalog.pending_deletion.clear();Ok(())
        })
    }

    pub fn list_sessions(&self, query: &SessionQuery) -> Result<Vec<SessionSummary>> {
        self.with_catalog(false, |catalog| {
            if let Some(id) = &query.project_id { ensure_project(catalog, id)?; }
            let search = query.search.trim().to_lowercase();
            let pin_order: HashMap<_, _> = catalog.pinned_order.iter().enumerate().map(|(index,id)|(id.as_str(),index)).collect();
            let mut sessions: Vec<_> = catalog.sessions.iter().filter(|session| {
                if query.project_id.as_deref().is_some_and(|id| session.project_id.as_deref() != Some(id)) { return false; }
                if matches!(query.archive, ArchiveFilter::Active) && session.archived { return false; }
                if matches!(query.archive, ArchiveFilter::Archived) && !session.archived { return false; }
                if query.pinned.is_some_and(|pinned| session.pinned != pinned) { return false; }
                if session.bot.is_some() { return false; }
                if query.hide_empty && session.last_active.is_none() && matches!(session.source, SessionSource::Native) { return false; }
                search.is_empty() || session.title.to_lowercase().contains(&search) || session.id.to_lowercase().contains(&search)
                    || matches!(&session.source, SessionSource::Hermes {session_id,..} if session_id.to_lowercase().contains(&search))
            }).map(|session| self.summary(session)).collect();
            sessions.sort_by(|a,b| b.pinned.cmp(&a.pinned)
                .then_with(|| if a.pinned && b.pinned { pin_order.get(a.id.as_str()).copied().unwrap_or(usize::MAX).cmp(&pin_order.get(b.id.as_str()).copied().unwrap_or(usize::MAX)) } else { std::cmp::Ordering::Equal })
                .then_with(|| b.recency().cmp(a.recency())).then_with(|| b.created_at.cmp(&a.created_at)).then_with(|| a.id.cmp(&b.id)));
            Ok(sessions)
        })
    }

    /// Soft-hide/restore. No engine session is opened and no transcript is
    /// rewritten, including when another window currently owns the engine lock.
    pub fn set_archived(&self, id: &str, archived: bool) -> Result<SessionSummary> {
        self.with_catalog(true, |catalog| {
            let record = find_session_mut(catalog, id)?;
            record.archived = archived;
            let summary = self.summary(record);
            if archived && catalog.active_session.as_deref() == Some(id) {
                catalog.active_session = None;
            }
            Ok(summary)
        })
    }

    /// Pin order is append order, as in the original desktop's pinSession().
    /// Unpinning/re-pinning moves a session to the end without changing its
    /// conversation activity timestamp or the transcript.
    pub fn set_pinned(&self, id: &str, pinned: bool) -> Result<SessionSummary> {
        self.with_catalog(true, |catalog| {
            let record = find_session_mut(catalog, id)?;
            record.pinned = pinned;
            let summary = self.summary(record);
            if pinned {
                if !catalog.pinned_order.iter().any(|value| value == id) {
                    catalog.pinned_order.push(id.to_owned());
                }
            } else {
                catalog.pinned_order.retain(|value| value != id);
            }
            Ok(summary)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hermes_engine::{Session, Status, Turn};

    #[test]
    fn archive_and_restore_survive_restart_without_rewriting_locked_transcript() {
        let temp = tempfile::tempdir().unwrap();
        let library = Library::open(temp.path()).unwrap();
        let summary = library
            .create_session(None, Some("A durable conversation"))
            .unwrap();
        let mut history = History::default();
        history.turns.push(Turn { context: String::new(),
            attachments: vec![],
            prompt: "Synthetic original prompt".into(),
            model: "fixture".into(),
            entries: vec![],
            status: Status::Complete,
        });
        atomic_json(&summary.directory.join("conversation.json"), &history).unwrap();
        let owned = Session::open(&summary.directory).unwrap();
        let before = fs::read(summary.directory.join("conversation.json")).unwrap();
        library.set_pinned(&summary.id, true).unwrap();
        library.set_archived(&summary.id, true).unwrap();
        assert!(library.active_session().unwrap().is_none());
        assert!(library.sessions(None).unwrap().is_empty());
        let reopened = Library::open(temp.path()).unwrap();
        let archived = reopened
            .list_sessions(&SessionQuery {
                archive: ArchiveFilter::Archived,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(archived.len(), 1);
        assert!(archived[0].archived && archived[0].pinned);
        assert_eq!(
            fs::read(summary.directory.join("conversation.json")).unwrap(),
            before
        );
        assert_eq!(owned.history(), &history);
        // Looking up/opening an intentionally archived chat does not silently
        // undo the user's archive (matches v0.23.3 manual archive provenance).
        reopened.select_session(&summary.id).unwrap();
        assert!(reopened.session(&summary.id).unwrap().archived);
        reopened.set_archived(&summary.id, false).unwrap();
        assert_eq!(reopened.sessions(None).unwrap().len(), 1);
        assert!(
            reopened
                .list_sessions(&SessionQuery {
                    archive: ArchiveFilter::Archived,
                    ..Default::default()
                })
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            reopened.session(&summary.id).unwrap().updated_at,
            summary.updated_at
        );
        assert_eq!(
            fs::read(summary.directory.join("conversation.json")).unwrap(),
            before
        );
    }

    #[test]
    fn pin_order_is_durable_and_unpin_does_not_fabricate_recent_activity() {
        let temp = tempfile::tempdir().unwrap();
        let library = Library::open(temp.path()).unwrap();
        let a = library.create_session(None, Some("First")).unwrap();
        let b = library.create_session(None, Some("Second")).unwrap();
        let c = library.create_session(None, Some("Third")).unwrap();
        library.set_pinned(&a.id, true).unwrap();
        library.set_pinned(&b.id, true).unwrap();
        library.set_pinned(&a.id, true).unwrap();
        let reopened = Library::open(temp.path()).unwrap();
        let pinned = SessionQuery {
            pinned: Some(true),
            ..Default::default()
        };
        assert_eq!(
            reopened
                .list_sessions(&pinned)
                .unwrap()
                .iter()
                .map(|s| s.id.as_str())
                .collect::<Vec<_>>(),
            [a.id.as_str(), b.id.as_str()]
        );
        reopened.set_pinned(&a.id, false).unwrap();
        reopened.set_pinned(&a.id, true).unwrap();
        assert_eq!(
            reopened
                .list_sessions(&pinned)
                .unwrap()
                .iter()
                .map(|s| s.id.as_str())
                .collect::<Vec<_>>(),
            [b.id.as_str(), a.id.as_str()]
        );
        assert_eq!(reopened.session(&a.id).unwrap().updated_at, a.updated_at);
        assert!(!reopened.session(&c.id).unwrap().pinned);
    }

    #[test]
    fn title_id_origin_and_archive_filters_compose_with_workspace() {
        let temp = tempfile::tempdir().unwrap();
        let library = Library::open(temp.path().join("library")).unwrap();
        let folder = temp.path().join("project");
        fs::create_dir(&folder).unwrap();
        let project = library.add_project(&folder).unwrap();
        let archived = library
            .create_session(Some(&project.id), Some("Release PLAN"))
            .unwrap();
        let active = library.create_session(None, Some("Release notes")).unwrap();
        library
            .with_catalog(true, |catalog| {
                find_session_mut(catalog, &archived.id)?.source = SessionSource::Hermes {
                    database: temp.path().join("synthetic.db"),
                    session_id: "Hermes-ORIGIN-42".into(),
                    imported_at: now(),
                };
                Ok(())
            })
            .unwrap();
        library.set_archived(&archived.id, true).unwrap();
        assert_eq!(
            library
                .list_sessions(&SessionQuery {
                    search: "release".into(),
                    ..Default::default()
                })
                .unwrap()[0]
                .id,
            active.id
        );
        for search in [&archived.id, "origin-42", "PLAN"] {
            let result = library
                .list_sessions(&SessionQuery {
                    project_id: Some(project.id.clone()),
                    archive: ArchiveFilter::Archived,
                    search: search.to_string(),
                    ..Default::default()
                })
                .unwrap();
            assert_eq!(result.len(), 1);
            assert_eq!(result[0].id, archived.id);
        }
        assert_eq!(
            library
                .list_sessions(&SessionQuery {
                    archive: ArchiveFilter::All,
                    search: "release".into(),
                    ..Default::default()
                })
                .unwrap()
                .len(),
            2
        );
        assert!(library.set_archived("unknown", true).is_err());
        assert!(library.set_pinned("unknown", true).is_err());
    }

    #[test]
    fn old_catalogs_upgrade_additively_and_project_selection_skips_archive() {
        let temp = tempfile::tempdir().unwrap();
        let library = Library::open(temp.path()).unwrap();
        let a = library
            .create_session(None, Some("Existing version one"))
            .unwrap();
        let b = library.create_session(None, Some("Archive me")).unwrap();
        let path = temp.path().join("library.json");
        let mut json: serde_json::Value =
            serde_json::from_reader(File::open(&path).unwrap()).unwrap();
        json.as_object_mut().unwrap().remove("pinned_order");
        for row in json["sessions"].as_array_mut().unwrap() {
            row.as_object_mut().unwrap().remove("archived");
            row.as_object_mut().unwrap().remove("pinned");
        }
        atomic_json(&path, &json).unwrap();
        let reopened = Library::open(temp.path()).unwrap();
        assert!(!reopened.session(&a.id).unwrap().archived);
        assert!(!reopened.session(&a.id).unwrap().pinned);
        reopened.set_archived(&b.id, true).unwrap();
        reopened.select_project(None).unwrap();
        assert_eq!(reopened.active_session().unwrap().unwrap().id, a.id);
    }
}
