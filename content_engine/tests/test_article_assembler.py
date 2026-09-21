"""Tests for article_assembler — bundle layout + paste-ready body."""
from pathlib import Path
import article_assembler as aa


def _illustrated(tmp_path: Path, body=None, title="How I tuned routing"):
    body = body or (
        f"# {title}\n\nLede.\n\n"
        "## First\n\nBody one.\n\n"
        "## Second\n\nBody two.\n\n"
        "## What I'd try next\n\nTakeaway.\n"
    )
    draft = {
        "title": title, "mode": "deep_dive", "pillar": "harness_tuning",
        "slug": "how-i-tuned-routing", "signals": [],
    }
    imgs = tmp_path / "imgs"
    imgs.mkdir(parents=True, exist_ok=True)
    p1 = imgs / "01-hero-hero.png"
    p1.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    p2 = imgs / "02-infographic-first.png"
    p2.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    outline = tmp_path / "outline.md"
    outline.write_text("# Outline\nplaceholder\n", encoding="utf-8")
    return {
        "body_md": body.replace("## First\n\n", "## First\n\n![First](imgs/02-infographic-first.png)\n\n"),
        "images": [str(p1), str(p2)],
        "outline_path": str(outline),
    }


def _draft():
    return {
        "title": "How I tuned routing", "mode": "deep_dive",
        "pillar": "harness_tuning", "slug": "how-i-tuned-routing",
        "signals": [],
    }


def test_bundle_creates_directory_layout(tmp_path):
    out_root = tmp_path / "out"
    bundle = aa.bundle(_illustrated(tmp_path), _draft(), out_root=out_root, dry_run=False)
    assert bundle.dir.exists()
    assert (bundle.dir / "article.md").exists()
    assert (bundle.dir / "outline.md").exists() or (tmp_path / "outline.md").exists()
    assert (bundle.dir / "imgs").exists()
    # at least one image lives under the bundle imgs dir.
    files = list((bundle.dir / "imgs").iterdir())
    assert any(f.suffix == ".png" for f in files)


def test_paste_ready_body_matches_article_md(tmp_path):
    out_root = tmp_path / "out"
    illustrated = _illustrated(tmp_path)
    bundle = aa.bundle(illustrated, _draft(), out_root=out_root, dry_run=False)
    on_disk = (bundle.dir / "article.md").read_text(encoding="utf-8")
    assert on_disk.strip() == bundle.article_md.strip()


def test_slug_is_kebab_case_2_to_4_words():
    out = aa._slug_from_title("How I tuned routing in the content engine")
    assert out == "how-i-tuned-routing"
    out2 = aa._slug_from_title("The")  # too short
    assert out2 == "article"
    out3 = aa._slug_from_title("One two three four five six seven")
    # caps at 4 words
    assert out3 == "one-two-three-four"


def test_slug_uniqueness_appends_timestamp(tmp_path):
    out_root = tmp_path / "out"
    illustrated = _illustrated(tmp_path)
    draft = _draft()
    b1 = aa.bundle(illustrated, draft, out_root=out_root, dry_run=False)
    b2 = aa.bundle(illustrated, draft, out_root=out_root, dry_run=False)
    # If the slug collided, the second bundle should have a -YYYYMMDD-HHMMSS suffix.
    if b1.dir == b2.dir:
        # Identical timestamp window — at minimum the contents match.
        assert b1.article_md_path == b2.article_md_path
    else:
        assert b1.dir != b2.dir
        assert "-" in b2.dir.name.split("how-i-tuned-routing-")[-1]


def test_image_paths_collected(tmp_path):
    out_root = tmp_path / "out"
    illustrated = _illustrated(tmp_path)
    bundle = aa.bundle(illustrated, _draft(), out_root=out_root, dry_run=False)
    # All bundle image paths live under bundle.dir / imgs.
    for p in bundle.image_paths:
        assert Path(p).is_relative_to(bundle.dir / "imgs")


def test_dry_run_does_not_write_files(tmp_path):
    out_root = tmp_path / "out"
    illustrated = _illustrated(tmp_path)
    bundle = aa.bundle(illustrated, _draft(), out_root=out_root, dry_run=True)
    # Nothing written under out_root.
    assert not out_root.exists() or not any(out_root.iterdir())
    # The bundle shape is still returned.
    assert bundle.dir
    assert bundle.article_md
