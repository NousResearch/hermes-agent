use super::*;
use std::path::PathBuf;

fn artifact(path: &str, base: PathBuf, kind: Kind) -> Artifact {
    Artifact {
        id: format!("fixture:{path}"),
        kind,
        value: path.into(),
        label: "Synthetic output".into(),
        session_id: "fixture".into(),
        session_title: "Fixture".into(),
        timestamp: 1700000000000.0,
        base_dir: base,
    }
}

#[test]
fn explicit_preview_reads_originating_file_and_leaves_original_bytes_unchanged() {
    let temp = tempfile::tempdir().unwrap();
    let first = temp.path().join("first");
    let second = temp.path().join("second");
    std::fs::create_dir(&first).unwrap();
    std::fs::create_dir(&second).unwrap();
    std::fs::write(first.join("report.md"), "Original first-project output").unwrap();
    std::fs::write(second.join("report.md"), "Other project's file").unwrap();
    let item = artifact("./report.md", first.clone(), Kind::File);
    let result = preview(&item).unwrap();
    assert!(matches!(result,Preview::Text{text,..}if text=="Original first-project output"));
    assert_eq!(
        std::fs::read_to_string(first.join("report.md")).unwrap(),
        "Original first-project output"
    );
    assert!(preview(&artifact("./missing.pdf", first, Kind::File)).is_err());
    assert!(
        preview(&artifact(
            "https://example.com/report.pdf",
            second,
            Kind::File
        ))
        .is_err()
    );
}

#[test]
fn preview_bounds_binary_files_and_decodes_real_image_bytes() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(
        temp.path().join("large.txt"),
        vec![b'x'; TEXT_PREVIEW_BYTES + 100],
    )
    .unwrap();
    assert!(matches!(
        preview(&artifact("./large.txt", temp.path().into(), Kind::File)).unwrap(),
        Preview::Text {
            truncated: true,
            ..
        }
    ));
    std::fs::write(temp.path().join("binary.pdf"), [b'P', 0, b'F']).unwrap();
    assert!(matches!(
        preview(&artifact("./binary.pdf", temp.path().into(), Kind::File)).unwrap(),
        Preview::Binary { bytes: 3 }
    ));
    let image = ::image::RgbaImage::from_pixel(2, 3, ::image::Rgba([20, 40, 60, 255]));
    image.save(temp.path().join("image.png")).unwrap();
    assert!(matches!(
        preview(&artifact("./image.png", temp.path().into(), Kind::Image)).unwrap(),
        Preview::Image {
            width: 2,
            height: 3,
            ..
        }
    ));
}

#[test]
fn pagination_matches_original_small_and_ellipsis_ranges() {
    assert_eq!(
        pagination_items(1, 5),
        [Some(1), Some(2), Some(3), Some(4), Some(5)]
    );
    assert_eq!(
        pagination_items(5, 12),
        [Some(1), None, Some(4), Some(5), Some(6), None, Some(12)]
    );
    assert_eq!(current_page(99, 25, 24), 2);
    assert_eq!(current_page(0, 0, 100), 1);
}

#[test]
fn opening_web_artifact_requires_an_explicit_action_and_stale_preview_cannot_reopen() {
    let item = artifact("https://example.com/report", PathBuf::new(), Kind::Link);
    let id = item.id.clone();
    let mut state = Artifacts {
        index: Some(Arc::new(Index {
            artifacts: vec![item],
            ..Default::default()
        })),
        ..Default::default()
    };
    let (_, effect) = state.update(Message::Search("report".into()), None, None);
    assert!(effect.is_none());
    let (_, effect) = state.update(Message::Open(id), None, None);
    assert!(matches!(effect,Some(Effect::OpenLink(url))if url=="https://example.com/report"));
    state.preview_id = Some("old".into());
    state.preview_generation = 4;
    state.close_preview();
    let (_, effect) = state.update(
        Message::Previewed(4, "old".into(), Ok(Arc::new(Preview::Binary { bytes: 1 }))),
        None,
        None,
    );
    assert!(effect.is_none());
    assert!(state.preview.is_none());
    assert!(state.preview_id.is_none());
    assert_eq!(
        format!("{:?}", Message::Search("private fixture".into())),
        "SearchArtifacts"
    );
}
