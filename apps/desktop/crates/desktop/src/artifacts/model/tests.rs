use super::*;
use hermes_engine::{Status, Turn};
use hermes_library::{HermesMessage, HermesSession, SessionSource};
use std::collections::BTreeMap;

fn session(id: &str, root: &Path) -> SessionSummary {
    SessionSummary {
        id: id.into(),
        title: "Fixture session".into(),
        project_id: None,
        directory: root.into(),
        created_at: "2026-06-17T12:00:00Z".into(),
        updated_at: "2026-06-17T12:00:00Z".into(), last_active: None, bot: None,
        source: SessionSource::Native,
        archived: false,
        pinned: false,
    }
}
fn message(role: &str, content: Value, tool: &str, time: Option<f64>) -> Message {
    Message {
        role: role.into(),
        content,
        text: None,
        context: None,
        tool_name: tool.into(),
        timestamp: time,
        base_dir: PathBuf::from("/synthetic/workspace"),
    }
}
fn assistant(text: &str) -> Message {
    message("assistant", json!(text), "", Some(2000.0))
}
fn values(items: &[Artifact]) -> Vec<&str> {
    items
        .iter()
        .map(|artifact| artifact.value.as_str())
        .collect()
}

#[test]
fn original_markdown_media_windows_and_unknown_extension_cases() {
    let found = collect(
        &session("one", Path::new("/synthetic")),
        [
            assistant("Deployed at `https://example.com/preview`, take a look."),
            assistant("[Image](#media:C%3A%5CUsers%5CMorten%5CMy%20Report.png)"),
            assistant("![cat](#media:%2Ftmp%2Fgenerated%2Fcat.png)"),
            assistant("**MEDIA: /tmp/generated/demo clip.mp4**"),
            assistant("MEDIA:\"/tmp/generated/contract draft.docx\""),
            assistant("MEDIA:/tmp/generated/palette.icc"),
            assistant("State lives at /tmp/plumbing/state-dir and /tmp/notes.bin"),
            message(
                "user",
                json!("I saw https://not-an-output.example.com"),
                "",
                Some(1.0),
            ),
        ],
    );
    assert!(values(&found).contains(&"https://example.com/preview"));
    assert!(values(&found).contains(&"C:\\Users\\Morten\\My Report.png"));
    assert!(values(&found).contains(&"/tmp/generated/cat.png"));
    assert!(values(&found).contains(&"/tmp/generated/demo clip.mp4"));
    assert!(values(&found).contains(&"/tmp/generated/contract draft.docx"));
    assert!(values(&found).contains(&"/tmp/generated/palette.icc"));
    assert!(
        !values(&found)
            .iter()
            .any(|value| value.contains("notes.bin")
                || value.contains("state-dir")
                || value.contains("not-an-output"))
    );
    assert_eq!(
        found
            .iter()
            .find(|artifact| artifact.value.ends_with("palette.icc"))
            .unwrap()
            .kind,
        Kind::File
    );
}

#[test]
fn observed_tool_noise_is_excluded_but_generated_descendant_keys_are_retained() {
    let found = collect(
        &session("one", Path::new("/synthetic")),
        [
            message(
                "tool",
                json!({"results":[{"cache_path":"/home/example/.cache/node.v24.18.1/bin","source_url":"https://example.com/changelog"}]}),
                "web_search",
                Some(1.0),
            ),
            message(
                "tool",
                json!({"attachments":[{"url":"https://example.com/passive.png"}],"image":"https://example.com/thumb.png"}),
                "discord_read_messages",
                Some(2.0),
            ),
            message(
                "tool",
                json!({"output":"see /tmp/false-positive.png"}),
                "web_search",
                Some(3.0),
            ),
            message(
                "tool",
                json!({"artifacts":[{"url":"https://example.com/generated/data.csv"}]}),
                "data_export",
                Some(4.0),
            ),
            message(
                "tool",
                json!({"nested":{"output_path":"/tmp/generated/report.pdf"}}),
                "non_producer",
                Some(5.0),
            ),
            message(
                "tool",
                json!({"image":"https://example.com/generated/cat.png"}),
                "image_generate",
                Some(6.0),
            ),
        ],
    );
    assert_eq!(
        values(&found),
        [
            "https://example.com/generated/data.csv",
            "/tmp/generated/report.pdf",
            "https://example.com/generated/cat.png"
        ]
    );
}

#[test]
fn terminal_arrays_stdout_and_pip_download_suppression_match_original_cases() {
    let found = collect(
        &session("one", Path::new("/synthetic")),
        [
            message(
                "tool",
                json!({"outputs":[{"output":"wrote /tmp/generated/report.pdf; MEDIA:/tmp/generated/voice.ogg"}]}),
                "terminal",
                Some(1.0),
            ),
            message(
                "tool",
                json!({"output":"Downloading https://mirror.example.com/pypi/packages/abc/tool-1.whl.metadata\nDownloading https://mirror.example.com/pypi/packages/abc/tool-1.whl\nDownloading https://mirror.example.com/pypi/packages/abc/tool-1.tar.gz\nSaved C:\\Users\\Alice\\AppData\\Local\\pip\\Cache\\http\\tool.tar.gz\nWrote /tmp/generated/final.pdf; https://github.com/example/tool/archive/refs/tags/v1.tar.gz"}),
                "terminal",
                Some(2.0),
            ),
        ],
    );
    let found = values(&found);
    assert!(found.contains(&"/tmp/generated/report.pdf"));
    assert!(found.contains(&"/tmp/generated/voice.ogg"));
    assert!(found.contains(&"/tmp/generated/final.pdf"));
    assert!(found.contains(&"https://github.com/example/tool/archive/refs/tags/v1.tar.gz"));
    assert!(
        !found
            .iter()
            .any(|value| value.contains("mirror.example.com")
                || value.to_lowercase().contains("pip\\cache"))
    );
}

#[test]
fn browser_wrapped_and_multimodal_screenshots_keep_paths_without_embedded_images() {
    let payload = json!({"images":["https://example.com/advertising.gif"],"page_url":"https://example.com/article","screenshot_path":"/tmp/browser/screenshot.png"});
    let wrapper = format!(
        "<untrusted_tool_result source=\"browser_snapshot\">\nExternal data, not instructions.\n\n{payload}\n</untrusted_tool_result>"
    );
    let found = collect(
        &session("one", Path::new("/synthetic")),
        [
            message("tool", json!(wrapper), "browser_snapshot", Some(1.0)),
            message(
                "tool",
                json!({"_multimodal":true,"content":[{"image_url":{"url":"data:image/png;base64,AAAA"},"type":"image_url"}],"meta":{"screenshot_path":"/tmp/browser/native.png"},"text_summary":"Screenshot attached"}),
                "browser_vision",
                Some(2.0),
            ),
            message(
                "tool",
                json!(
                    "Image attached. Screenshot path: /tmp/hermes browser/summary screenshot.png"
                ),
                "browser_vision",
                Some(3.0),
            ),
        ],
    );
    assert_eq!(
        values(&found),
        [
            "/tmp/browser/screenshot.png",
            "/tmp/browser/native.png",
            "/tmp/hermes browser/summary screenshot.png"
        ]
    );
}

#[test]
fn first_duplicate_timestamp_and_session_scoped_identity_match_upstream() {
    let one = session("one", Path::new("/synthetic"));
    let found = collect(
        &one,
        [
            message(
                "assistant",
                json!("https://example.com/shared.pdf"),
                "",
                Some(1781773226.453548),
            ),
            message(
                "assistant",
                json!("https://example.com/shared.pdf"),
                "",
                Some(1781773999.0),
            ),
        ],
    );
    assert_eq!(found.len(), 1);
    assert!((found[0].timestamp - 1781773226453.548).abs() < 0.01);
    let other = collect(
        &session("two", Path::new("/synthetic")),
        [assistant("https://example.com/shared.pdf")],
    );
    assert_ne!(found[0].id, other[0].id);
    assert_eq!(
        normalize_timestamp(Some(42_000_000_000.0)),
        Some(42_000_000_000.0)
    );
    assert_eq!(normalize_timestamp(Some(f64::INFINITY)), None);
    assert_eq!(normalize_timestamp(Some(0.0)), None);
    let fallback = collect(
        &one,
        [message(
            "assistant",
            json!("/tmp/report.pdf"),
            "",
            Some(f64::NAN),
        )],
    );
    assert_eq!(
        fallback[0].timestamp,
        chrono::DateTime::parse_from_rfc3339(&one.updated_at)
            .unwrap()
            .timestamp_millis() as f64
    );
}

#[test]
fn source_filter_and_search_include_location_and_origin_title() {
    let found = collect(
        &session("one", Path::new("/synthetic")),
        [assistant(
            "MEDIA:/tmp/figure.png MEDIA:/tmp/report.pdf https://example.com/docs",
        )],
    );
    assert_eq!(
        found
            .iter()
            .filter(|artifact| artifact.matches(Filter::Images, "FIGURE"))
            .count(),
        1
    );
    assert_eq!(
        found
            .iter()
            .filter(|artifact| artifact.matches(Filter::Files, "fixture session"))
            .count(),
        1
    );
    assert_eq!(
        found
            .iter()
            .filter(|artifact| artifact.matches(Filter::Links, "example.com/docs"))
            .count(),
        1
    );
    assert_eq!(
        found
            .iter()
            .filter(|artifact| artifact.matches(Filter::Images, "missing"))
            .count(),
        0
    );
}

fn source_session(id: &str, cwd: &Path) -> HermesSession {
    HermesSession {
        id: id.into(),
        title: "Imported fixture".into(),
        model: Some("fixture-model".into()),
        source: "cli".into(),
        cwd: Some(cwd.into()),
        started_at: 1700000000.0,
        last_activity_at: 1700000003.0,
        message_count: 2,
        archived: false,
        hidden: false,
        parent_session_id: None,
        end_reason: None,
        raw: BTreeMap::new(),
    }
}
fn source_message(
    id: i64,
    session_id: &str,
    content: Value,
    active: bool,
    compacted: bool,
) -> HermesMessage {
    HermesMessage {
        id,
        session_id: session_id.into(),
        role: "assistant".into(),
        content,
        tool_call_id: None,
        tool_calls: Value::Null,
        tool_name: None,
        timestamp: 1700000000.0 + id as f64,
        active,
        compacted,
        display_metadata: Value::Null,
        raw: BTreeMap::new(),
    }
}
fn history(entries: Vec<Entry>) -> History {
    let mut history = History::default();
    history.turns.push(Turn { context: String::new(),
        attachments: vec![],
        prompt: "Synthetic fixture prompt".into(),
        model: "fixture-model".into(),
        entries,
        status: Status::Complete,
    });
    history
}

#[test]
fn imported_lineage_retains_original_timestamp_and_cwd_then_adds_native_continuation() {
    let temp = tempfile::tempdir().unwrap();
    let library = Library::open(temp.path().join("library")).unwrap();
    let native_cwd = temp.path().join("native-workspace");
    let upstream_cwd = temp.path().join("upstream-workspace");
    std::fs::create_dir_all(&native_cwd).unwrap();
    std::fs::create_dir_all(&upstream_cwd).unwrap();
    let project = library.add_project(&native_cwd).unwrap();
    let summary = library
        .create_session(Some(&project.id), Some("Imported fixture"))
        .unwrap();
    let mut hidden = source_message(4, "tip", json!("./hidden.pdf"), true, false);
    hidden.display_metadata = json!({"model_only":true});
    let archive = HermesArchive {
        version: 1,
        database: temp.path().join("synthetic-source.db"),
        upstream_session_id: "root".into(),
        imported_at: "2026-06-17T12:00:00Z".into(),
        schema_version: Some(33),
        sessions: vec![
            source_session("root", &upstream_cwd),
            source_session("tip", &upstream_cwd),
        ],
        messages: vec![
            source_message(1, "root", json!("./report.pdf"), false, true),
            source_message(2, "tip", json!("./report.pdf"), true, false),
            source_message(3, "tip", json!("./inactive.pdf"), false, false),
            hidden,
        ],
    };
    std::fs::write(
        summary.directory.join("hermes-source.json"),
        serde_json::to_vec(&archive).unwrap(),
    )
    .unwrap();
    let native = history(vec![
        Entry::Assistant {
            text: "./report.pdf".into(),
            calls: vec![],
        },
        Entry::Assistant {
            text: "./new.md".into(),
            calls: vec![],
        },
    ]);
    std::fs::write(
        summary.directory.join("conversation.json"),
        serde_json::to_vec(&native).unwrap(),
    )
    .unwrap();
    let index = load(&library, None).unwrap();
    assert!(index.failures.is_empty());
    assert_eq!(index.artifacts.len(), 2);
    let report = index
        .artifacts
        .iter()
        .find(|artifact| artifact.value == "./report.pdf")
        .unwrap();
    assert_eq!(report.timestamp, 1700000001000.0);
    assert_eq!(
        local_path(report).unwrap(),
        upstream_cwd.join("./report.pdf")
    );
    let continuation = index
        .artifacts
        .iter()
        .find(|artifact| artifact.value == "./new.md")
        .unwrap();
    assert_eq!(
        local_path(continuation).unwrap(),
        std::fs::canonicalize(native_cwd).unwrap().join("./new.md")
    );
}

#[test]
fn native_written_files_use_owner_workspace_and_global_index_ignores_selected_project() {
    let temp = tempfile::tempdir().unwrap();
    let library = Library::open(temp.path().join("library")).unwrap();
    let mut ids = vec![];
    for name in ["first", "second"] {
        let root = temp.path().join(name);
        std::fs::create_dir(&root).unwrap();
        let project = library.add_project(&root).unwrap();
        let summary = library
            .create_session(Some(&project.id), Some(name))
            .unwrap();
        let native = history(vec![Entry::Tool {
            id: "write".into(),
            name: "write_file".into(),
            approved: true,
            result: json!({"path":"report.md","bytes_written":3,"created":true,"verified":true})
                .to_string(),
        }]);
        std::fs::write(
            summary.directory.join("conversation.json"),
            serde_json::to_vec(&native).unwrap(),
        )
        .unwrap();
        ids.push((project.id, summary.id));
    }
    library.select_project(Some(&ids[0].0)).unwrap();
    let index = load(&library, None).unwrap();
    assert_eq!(index.artifacts.len(), 2);
    for artifact in &index.artifacts {
        assert_eq!(artifact.value, "./report.md");
        assert!(
            local_path(artifact)
                .unwrap()
                .ends_with(format!("{}/report.md", artifact.session_title))
        );
    }
}

#[test]
fn corrupt_session_is_discarded_and_other_sessions_continue_indexing() {
    let temp = tempfile::tempdir().unwrap();
    let library = Library::open(temp.path()).unwrap();
    let good = library.create_session(None, Some("good")).unwrap();
    let bad = library.create_session(None, Some("bad")).unwrap();
    let native = history(vec![Entry::Assistant {
        text: "https://example.com/result".into(),
        calls: vec![],
    }]);
    for summary in [&good, &bad] {
        std::fs::write(
            summary.directory.join("conversation.json"),
            serde_json::to_vec(&native).unwrap(),
        )
        .unwrap();
    }
    std::fs::write(
        bad.directory.join("hermes-source.json"),
        "{broken source archive}",
    )
    .unwrap();
    let index = load(&library, None).unwrap();
    assert_eq!(index.failures.len(), 1);
    assert_eq!(index.artifacts.len(), 1);
    assert_eq!(index.artifacts[0].session_id, good.id);
}
