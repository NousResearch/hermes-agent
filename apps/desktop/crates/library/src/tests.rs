use super::*;
use hermes_engine::{Entry, Session, Status, Turn};
use rusqlite::{Connection, params};
use serde_json::json;

// Synthetic fixture follows the real upstream sessions/messages table and its
// sentinel format. Tests never discover or open the user's ~/.hermes files.
const SCHEMA: &str = r#"
CREATE TABLE schema_version (version INTEGER NOT NULL);
INSERT INTO schema_version VALUES (33);
CREATE TABLE sessions (
    id TEXT PRIMARY KEY, source TEXT NOT NULL, model TEXT, model_config TEXT,
    parent_session_id TEXT, started_at REAL NOT NULL, ended_at REAL, end_reason TEXT,
    title TEXT, cwd TEXT, last_activity_at REAL, message_count INTEGER DEFAULT 0,
    archived INTEGER DEFAULT 0, hidden INTEGER DEFAULT 0
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL, content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT,
    timestamp REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1,
    compacted INTEGER NOT NULL DEFAULT 0, _compressed_summary INTEGER NOT NULL DEFAULT 0,
    display_metadata TEXT, display_identity BLOB, display_order INTEGER, message_uid TEXT,
    reasoning TEXT
);
"#;

fn synthetic_database(path: &Path) -> Connection {
    let connection = Connection::open(path).unwrap();
    connection.execute_batch(SCHEMA).unwrap();
    connection
}

fn add_session(
    connection: &Connection,
    id: &str,
    title: &str,
    parent: Option<&str>,
    end_reason: Option<&str>,
    model_config: Option<&str>,
) {
    connection.execute("INSERT INTO sessions (id, source, model, model_config, parent_session_id, started_at, ended_at, end_reason, title, cwd, message_count) VALUES (?1, 'cli', 'test-model', ?2, ?3, 1700000000.0, ?4, ?5, ?6, '/never-execute-imported-cwd', 2)", params![id, model_config, parent, end_reason.map(|_| 1700000001.0), end_reason, title]).unwrap();
}

fn add_message(connection: &Connection, session: &str, role: &str, content: &str, timestamp: f64) {
    connection
        .execute(
            "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?1, ?2, ?3, ?4)",
            params![session, role, content, timestamp],
        )
        .unwrap();
}

#[test]
fn native_catalog_persists_workspace_selection_titles_and_distinct_histories() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace = temporary.path().join("project");
    fs::create_dir(&workspace).unwrap();
    let root = temporary.path().join("library");
    let library = Library::open(&root).unwrap();
    let project = library.add_project(&workspace).unwrap();
    assert_eq!(library.add_project(&workspace).unwrap(), project);
    library.select_project(Some(&project.id)).unwrap();
    let first = library.create_session(Some(&project.id), None).unwrap();
    let second = library.create_session(None, Some("Other chat")).unwrap();
    assert_ne!(first.id, second.id);
    assert_ne!(first.directory, second.directory);
    library.rename_session(&first.id, "Release notes").unwrap();
    library.select_session(&first.id).unwrap();
    let reopened = Library::open(root).unwrap();
    assert_eq!(reopened.selected_project().unwrap(), Some(project.clone()));
    assert_eq!(
        reopened.active_session().unwrap().unwrap().title,
        "Release notes"
    );
    assert_eq!(
        reopened.workspace_for_session(&first.id).unwrap(),
        Some(fs::canonicalize(workspace).unwrap())
    );
    assert_eq!(reopened.workspace_for_session(&second.id).unwrap(), None);
    assert_eq!(reopened.sessions(Some(&project.id)).unwrap().len(), 1);
    assert_eq!(reopened.sessions(None).unwrap().len(), 2);
    assert!(
        Session::open(&first.directory)
            .unwrap()
            .history()
            .turns
            .is_empty()
    );
    assert!(
        Session::open(&second.directory)
            .unwrap()
            .history()
            .turns
            .is_empty()
    );
}

#[test]
fn title_comes_from_first_prompt_and_user_rename_is_retained() {
    let temporary = tempfile::tempdir().unwrap();
    let library = Library::open(temporary.path()).unwrap();
    let session = library.create_session(None, None).unwrap();
    let mut history = History::default();
    history.turns.push(Turn { context: String::new(),
        attachments: vec![],
        prompt: "  Help\n with   release notes  ".to_owned(),
        model: "test".to_owned(),
        entries: vec![],
        status: Status::Complete,
    });
    atomic_json(&session.directory.join("conversation.json"), &history).unwrap();
    assert_eq!(
        library.touch_session(&session.id).unwrap().title,
        "Help with release notes"
    );
    library.rename_session(&session.id, "My title").unwrap();
    assert_eq!(
        library.touch_session(&session.id).unwrap().title,
        "My title"
    );
    assert!(library.rename_session(&session.id, "   ").is_err());
    assert!(
        library
            .rename_session(&session.id, &"a".repeat(201))
            .is_err()
    );
    assert!(library.create_session(Some("missing"), None).is_err());
    assert!(
        library
            .add_project(&temporary.path().join("missing"))
            .is_err()
    );
}

#[test]
fn legacy_rust_history_is_copied_once_and_original_is_untouched() {
    let temporary = tempfile::tempdir().unwrap();
    let mut history = History::default();
    history.turns.push(Turn { context: String::new(),
        attachments: vec![],
        prompt: "Legacy synthetic prompt".to_owned(),
        model: "test".to_owned(),
        entries: vec![],
        status: Status::Interrupted,
    });
    let original = temporary.path().join("conversation.json");
    atomic_json(&original, &history).unwrap();
    let before = fs::read(&original).unwrap();
    let library = Library::open(temporary.path()).unwrap();
    let sessions = library.sessions(None).unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].source, SessionSource::LegacyRust);
    assert_eq!(
        Session::open(&sessions[0].directory).unwrap().history(),
        &history
    );
    assert_eq!(fs::read(&original).unwrap(), before);
    assert_eq!(
        Library::open(temporary.path())
            .unwrap()
            .sessions(None)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn concurrent_catalog_writers_do_not_lose_sessions() {
    let temporary = tempfile::tempdir().unwrap();
    let library = Library::open(temporary.path()).unwrap();
    let mut threads = vec![];
    for number in 0..12 {
        let library = library.clone();
        threads.push(std::thread::spawn(move || {
            library
                .create_session(None, Some(&format!("Session {number}")))
                .unwrap()
                .id
        }));
    }
    let ids: HashSet<_> = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect();
    assert_eq!(ids.len(), 12);
    assert_eq!(library.sessions(None).unwrap().len(), 12);
}

#[test]
fn import_preserves_source_and_raw_content_and_resumes_native_copy() {
    let temporary = tempfile::tempdir().unwrap();
    let source_path = temporary.path().join("source.db");
    let connection = synthetic_database(&source_path);
    connection
        .execute_batch("ALTER TABLE sessions ADD COLUMN system_prompt TEXT;")
        .unwrap();
    add_session(
        &connection,
        "original-id",
        "Imported chat",
        None,
        None,
        None,
    );
    connection.execute("UPDATE sessions SET system_prompt='Synthetic original instruction' WHERE id='original-id'", []).unwrap();
    add_message(
        &connection,
        "original-id",
        "system",
        "Original system instruction",
        1.0,
    );
    let multimodal = format!(
        "\0json:{}",
        json!([{"type":"text","text":"Synthetic image question"},{"type":"image_url","image_url":{"url":"synthetic://fixture"}}])
    );
    add_message(&connection, "original-id", "user", &multimodal, 2.0);
    connection.execute("INSERT INTO messages(session_id,role,content,tool_calls,timestamp,reasoning) VALUES (?1,'assistant','Checking','[{\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"get_current_time\",\"arguments\":\"{}\"}}]',3.0,'synthetic reasoning')", ["original-id"]).unwrap();
    connection.execute("INSERT INTO messages(session_id,role,content,tool_call_id,tool_name,timestamp) VALUES ('original-id','tool','{\"utc\":\"synthetic\"}','call-1','get_current_time',4.0)", []).unwrap();
    add_message(
        &connection,
        "original-id",
        "assistant",
        "Recorded answer",
        5.0,
    );
    connection.execute("INSERT INTO messages(session_id,role,content,timestamp,active) VALUES ('original-id','user','Inactive original',6.0,0)", []).unwrap();
    drop(connection);
    let before = fs::read(&source_path).unwrap();
    let modified = fs::metadata(&source_path).unwrap().modified().unwrap();
    let source = HermesHistory::open(&source_path).unwrap();
    assert_eq!(source.sessions().unwrap()[0].id, "original-id");
    assert!(
        !source.sessions().unwrap()[0]
            .raw
            .contains_key("system_prompt")
    );
    let library = Library::open(temporary.path().join("native")).unwrap();
    let report = library.import_hermes(&source, "original-id", None).unwrap();
    assert!(!report.already_imported);
    assert_eq!(report.imported_messages, 5);
    assert_eq!(
        library.workspace_for_session(&report.session.id).unwrap(),
        None
    );
    let native = Session::open(&report.session.directory).unwrap();
    assert_eq!(native.history().turns.len(), 1);
    assert_eq!(native.history().turns[0].prompt, "Synthetic image question");
    assert_eq!(native.history().turns[0].status, Status::Complete);
    assert!(
        matches!(&native.history().turns[0].entries[1], Entry::Tool { approved: false, id, .. } if id == "call-1")
    );
    let archive: HermesArchive =
        serde_json::from_reader(File::open(&report.archive_path).unwrap()).unwrap();
    assert_eq!(archive.messages.len(), 6);
    assert_eq!(
        archive.sessions[0].raw.get("system_prompt"),
        Some(&json!("Synthetic original instruction"))
    );
    assert_eq!(
        archive.messages[2].raw.get("reasoning"),
        Some(&json!("synthetic reasoning"))
    );
    assert_eq!(archive.messages[1].content[1]["type"], "image_url");
    assert!(
        report
            .warnings
            .iter()
            .any(|warning| warning.contains("Image/audio"))
    );
    assert_eq!(fs::read(&source_path).unwrap(), before);
    assert_eq!(
        fs::metadata(&source_path).unwrap().modified().unwrap(),
        modified
    );
    assert!(!source_path.with_file_name("source.db-wal").exists());
    assert!(!source_path.with_file_name("source.db-shm").exists());
    let duplicate = library.import_hermes(&source, "original-id", None).unwrap();
    assert!(duplicate.already_imported);
    assert_eq!(duplicate.session.id, report.session.id);
    assert_eq!(library.sessions(None).unwrap().len(), 1);
}

#[test]
fn wal_snapshot_sees_committed_rows_without_touching_any_source_file() {
    let temporary = tempfile::tempdir().unwrap();
    let source_path = temporary.path().join("state.db");
    let connection = Connection::open(&source_path).unwrap();
    connection
        .execute_batch("PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0;")
        .unwrap();
    connection.execute_batch(SCHEMA).unwrap();
    add_session(
        &connection,
        "wal-session",
        "Committed in WAL",
        None,
        None,
        None,
    );
    add_message(
        &connection,
        "wal-session",
        "user",
        "A synthetic WAL turn",
        1.0,
    );
    let paths = [
        source_path.clone(),
        source_path.with_file_name("state.db-wal"),
        source_path.with_file_name("state.db-shm"),
    ];
    let before: Vec<_> = paths
        .iter()
        .map(|path| {
            (
                fs::read(path).unwrap(),
                fs::metadata(path).unwrap().modified().unwrap(),
            )
        })
        .collect();
    let source = HermesHistory::open(&source_path).unwrap();
    assert_eq!(source.sessions().unwrap().len(), 1);
    assert_eq!(
        source.read_session("wal-session").unwrap().messages[0].content,
        "A synthetic WAL turn"
    );
    for (path, (bytes, modified)) in paths.iter().zip(before) {
        assert_eq!(fs::read(path).unwrap(), bytes);
        assert_eq!(fs::metadata(path).unwrap().modified().unwrap(), modified);
    }
}

#[test]
fn compression_lineage_deduplicates_copied_messages_and_excludes_branches() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("source.db");
    let connection = synthetic_database(&path);
    add_session(
        &connection,
        "root",
        "Conversation",
        None,
        Some("compression"),
        None,
    );
    add_session(&connection, "tip", "Conversation", Some("root"), None, None);
    add_session(
        &connection,
        "branch",
        "Separate branch",
        Some("root"),
        None,
        Some(r#"{"_branched_from":"root"}"#),
    );
    add_message(&connection, "root", "user", "Original prompt", 1.0);
    connection.execute("UPDATE messages SET active=0, compacted=1, message_uid='same-uid' WHERE session_id='root'", []).unwrap();
    add_message(&connection, "tip", "user", "Original prompt", 1.0);
    connection
        .execute(
            "UPDATE messages SET message_uid='same-uid' WHERE session_id='tip'",
            [],
        )
        .unwrap();
    add_message(&connection, "tip", "assistant", "Original answer", 2.0);
    add_message(&connection, "branch", "user", "Should remain separate", 3.0);
    drop(connection);
    let source = HermesHistory::open(&path).unwrap();
    let archive = source.read_session("tip").unwrap();
    assert_eq!(
        archive
            .sessions
            .iter()
            .map(|session| session.id.as_str())
            .collect::<Vec<_>>(),
        ["root", "tip"]
    );
    assert_eq!(archive.messages.len(), 3);
    let library = Library::open(temporary.path().join("native")).unwrap();
    let report = library.import_hermes(&source, "root", None).unwrap();
    assert_eq!(report.imported_messages, 2);
    let native = Session::open(&report.session.directory).unwrap();
    assert_eq!(native.history().turns.len(), 1);
    assert_eq!(native.history().turns[0].prompt, "Original prompt");
    assert!(
        library
            .import_hermes(&source, "tip", None)
            .unwrap()
            .already_imported
    );
    assert_eq!(source.read_session("branch").unwrap().sessions.len(), 1);
}

#[test]
fn invalid_database_and_sql_injection_ids_are_read_only_failures() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("not-hermes.db");
    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch("CREATE TABLE unrelated(x);")
        .unwrap();
    drop(connection);
    let before = fs::read(&path).unwrap();
    assert!(HermesHistory::open(&path).is_err());
    assert_eq!(fs::read(&path).unwrap(), before);
    let valid = temporary.path().join("source.db");
    let connection = synthetic_database(&valid);
    add_session(&connection, "normal", "Test", None, None, None);
    drop(connection);
    let source = HermesHistory::open(&valid).unwrap();
    assert!(source.read_session("'; DROP TABLE sessions; --").is_err());
    assert_eq!(source.sessions().unwrap().len(), 1);
    assert!(!temporary.path().join("missing.db").exists());
    assert!(HermesHistory::open(&temporary.path().join("missing.db")).is_err());
    assert!(!temporary.path().join("missing.db").exists());
}

#[test]
fn older_optional_columns_and_literal_content_remain_compatible() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("older.db");
    let connection = Connection::open(&path).unwrap();
    connection.execute_batch("CREATE TABLE sessions(id TEXT PRIMARY KEY, source TEXT NOT NULL, started_at REAL NOT NULL); CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT, timestamp REAL NOT NULL); INSERT INTO sessions VALUES ('older', 'cli', 1.0);").unwrap();
    // Upstream wraps a literal reserved prefix as a JSON string; decoding must
    // return the literal once, rather than interpreting it as nested structure.
    let literal = "\0json:literal text";
    let encoded = format!("\0json:{}", serde_json::to_string(literal).unwrap());
    connection
        .execute(
            "INSERT INTO messages VALUES (1, 'older', 'user', ?1, 1.0)",
            [encoded],
        )
        .unwrap();
    connection.execute("INSERT INTO messages VALUES (2, 'older', 'assistant', 'An older recorded answer', 2.0)", []).unwrap();
    drop(connection);
    let source = HermesHistory::open(&path).unwrap();
    assert_eq!(source.sessions().unwrap()[0].title, "");
    let library = Library::open(temporary.path().join("native")).unwrap();
    let report = library.import_hermes(&source, "older", None).unwrap();
    let native = Session::open(&report.session.directory).unwrap();
    assert_eq!(native.history().turns[0].prompt, literal);
    assert_eq!(native.history().turns[0].status, Status::Complete);
    assert_eq!(report.imported_messages, 2);
}

#[test]
fn active_rollback_journal_is_rejected_without_recovery_or_source_writes() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("state.db");
    let connection = synthetic_database(&path);
    add_session(&connection, "synthetic", "Synthetic", None, None, None);
    drop(connection);
    let journal = path.with_file_name("state.db-journal");
    let journal_bytes = [0xd9, 0xd5, 0x05, 0xf9, 0x20, 0xa1, 0x63, 0xd7];
    fs::write(&journal, journal_bytes).unwrap();
    let before = fs::read(&path).unwrap();
    let error = HermesHistory::open(&path).err().unwrap().to_string();
    assert!(error.contains("rollback journal"));
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(fs::read(&journal).unwrap(), journal_bytes);
}

#[cfg(unix)]
#[test]
fn symlink_session_directories_are_rejected() {
    let temporary = tempfile::tempdir().unwrap();
    let library = Library::open(temporary.path().join("native")).unwrap();
    let session = library.create_session(None, None).unwrap();
    fs::remove_file(session.directory.join("conversation.json")).unwrap();
    fs::remove_dir(&session.directory).unwrap();
    std::os::unix::fs::symlink(temporary.path(), &session.directory).unwrap();
    assert!(library.session_dir(&session.id).is_err());
    assert!(library.select_session(&session.id).is_err());
}

fn send_turn(library: &Library, session: &SessionSummary, prompt: &str) {
    let mut history = History::default();
    history.turns.push(Turn { context: String::new(), attachments: vec![], prompt: prompt.to_owned(), model: "test".to_owned(), entries: vec![], status: Status::Complete });
    atomic_json(&session.directory.join("conversation.json"), &history).unwrap();
    library.touch_session(&session.id).unwrap();
}

#[test]
fn sidebar_order_follows_last_message_and_hides_empty_chats() {
    let temporary = tempfile::tempdir().unwrap();
    let library = Library::open(temporary.path()).unwrap();
    let sidebar = SessionQuery { hide_empty: true, ..Default::default() };
    let titles = |library: &Library| library.list_sessions(&sidebar).unwrap().into_iter().map(|s| s.title).collect::<Vec<_>>();

    let first = library.create_session(None, None).unwrap();
    send_turn(&library, &first, "first");
    let second = library.create_session(None, None).unwrap();
    assert_ne!(first.id, second.id);
    send_turn(&library, &second, "second");
    assert_eq!(titles(&library), ["second", "first"]);

    // Renaming, pinning-then-unpinning and archiving/restoring keep the order.
    library.rename_session(&first.id, "renamed").unwrap();
    library.set_pinned(&first.id, true).unwrap();
    library.set_pinned(&first.id, false).unwrap();
    library.set_archived(&first.id, true).unwrap();
    library.set_archived(&first.id, false).unwrap();
    assert_eq!(titles(&library), ["second", "renamed"]);

    // Opening "New session" repeatedly reuses one empty chat and never lists it.
    let empty = library.create_session(None, None).unwrap();
    assert_eq!(library.create_session(None, None).unwrap().id, empty.id);
    assert_eq!(titles(&library), ["second", "renamed"]);
    assert_eq!(library.sessions(None).unwrap().len(), 3);

    // Only a new message moves a chat to the top.
    send_turn(&library, &first, "again");
    assert_eq!(titles(&library), ["renamed", "second"]);
}

#[test]
fn existing_catalog_backfills_activity_once() {
    let temporary = tempfile::tempdir().unwrap();
    let library = Library::open(temporary.path()).unwrap();
    let used = library.create_session(None, None).unwrap();
    send_turn(&library, &used, "used");
    let mut catalog: serde_json::Value = serde_json::from_slice(&fs::read(temporary.path().join("library.json")).unwrap()).unwrap();
    catalog.as_object_mut().unwrap().remove("activity_migrated");
    for session in catalog["sessions"].as_array_mut().unwrap() { session.as_object_mut().unwrap().remove("last_active"); }
    // An old build also left an empty "New chat" behind.
    let empty_id = "chat-legacy-empty";
    let dir = temporary.path().join("sessions").join(empty_id);
    fs::create_dir(&dir).unwrap();
    atomic_json(&dir.join("conversation.json"), &History::default()).unwrap();
    let mut empty = catalog["sessions"][0].clone();
    empty["id"] = json!(empty_id); empty["title"] = json!("New chat");
    catalog["sessions"].as_array_mut().unwrap().push(empty);
    fs::write(temporary.path().join("library.json"), serde_json::to_vec(&catalog).unwrap()).unwrap();

    let reopened = Library::open(temporary.path()).unwrap();
    let visible = reopened.list_sessions(&SessionQuery { hide_empty: true, ..Default::default() }).unwrap();
    assert_eq!(visible.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), [used.id.as_str()]);
}
