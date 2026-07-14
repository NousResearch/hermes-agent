"""Hermetic tests for scripts/mw/page_check.py (MW-P3-I2b).

All fixtures live under tmp_path — no network, no browser, stdlib only.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# load module under test (path-stable; no package install required)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
PAGE_CHECK_PATH = REPO_ROOT / "scripts" / "mw" / "page_check.py"

_spec = importlib.util.spec_from_file_location("mw_page_check", PAGE_CHECK_PATH)
assert _spec and _spec.loader
pc = importlib.util.module_from_spec(_spec)
sys.modules["mw_page_check"] = pc
_spec.loader.exec_module(pc)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_bytes(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


GOOD_CONFIG = textwrap.dedent(
    """\
    lang:
      allowed: ["th", "en"]
      bilingual: false
      markers_th: ["ข่าว"]
      markers_en: ["News"]
    main_selector: "main"
    soft404:
      phrases: ["404", "ไม่พบหน้า", "Not Found"]
      min_content_chars: 50
    file_caps:
      hero_kb: 300
      content_kb: 150
      video_kb: 2048
      hero_selector: ".hero, [data-role=hero]"
    pagination:
      list_selector: ".list, [data-list]"
      item_selector: "li, .card"
      threshold: 12
      control_selector: ".pagination, nav[aria-label*=pag]"
    related:
      selector: ".related, [data-related]"
    video:
      require_muted_if_autoplay: true
      require_playsinline: true
      forbid_autoplay_with_audio: true
      require_reduced_motion_support: true
    checks:
      soft_404: {blocking: true}
      links_internal: {blocking: true}
      language: {blocking: true}
      file_size: {blocking: true}
      pagination: {blocking: true}
      video_attrs: {blocking: true}
      related_no_self: {blocking: true}
      sticky_cover: {blocking: false, manual: true}
    """
)

# Long enough main content for soft_404 min_content_chars=50
# NOTE: must NOT contain soft404 phrases like "404" / "Not Found" / "ไม่พบหน้า"
_MAIN_BODY = (
    "นี่คือเนื้อหาหลักของหน้าเว็บสำหรับทดสอบระบบตรวจหน้า "
    "มีข้อความยาวพอที่จะผ่านเกณฑ์ความยาวขั้นต่ำของ soft-not-found check."
)


def _good_html(
    *,
    lang: str = "th",
    main_extra: str = "",
    body_extra: str = "",
    head_extra: str = "",
    sticky: bool = False,
) -> str:
    sticky_block = (
        '<div class="header" style="position: sticky; top: 0">sticky bar</div>'
        if sticky
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
  <meta charset="utf-8"/>
  <title>Test Page</title>
  <link rel="canonical" href="https://example.com/page.html"/>
  {head_extra}
</head>
<body>
  {sticky_block}
  <main id="main">
    <p>{_MAIN_BODY}</p>
    {main_extra}
  </main>
  {body_extra}
</body>
</html>
"""


def _seed(
    tmp: Path,
    html: str,
    config: str = GOOD_CONFIG,
    *,
    assets: Optional[Dict[str, int]] = None,
) -> Tuple[Path, Path, Path]:
    """Create site dir with page.html, config, optional asset files (name→bytes)."""
    site = tmp / "site"
    site.mkdir(parents=True, exist_ok=True)
    html_path = _write(site / "page.html", html)
    cfg_path = _write(tmp / "page-check.yaml", config)
    if assets:
        for name, size in assets.items():
            _write_bytes(site / name, size)
    return html_path, cfg_path, site


def _run(
    html: Path,
    config: Path,
    *,
    asset_root: Optional[Path] = None,
    base_url: Optional[str] = None,
    as_json: bool = False,
) -> Tuple[int, str]:
    argv: List[str] = [str(html), "--config", str(config)]
    if asset_root is not None:
        argv += ["--asset-root", str(asset_root)]
    if base_url is not None:
        argv += ["--base-url", base_url]
    if as_json:
        argv.append("--json")
    # capture stdout
    import io
    from contextlib import redirect_stdout, redirect_stderr

    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = pc.main(argv)
    text = out.getvalue() + err.getvalue()
    return code, text


def _run_json(
    html: Path,
    config: Path,
    **kw: Any,
) -> Tuple[int, Dict[str, Any]]:
    code, text = _run(html, config, as_json=True, **kw)
    data = json.loads(text)
    return code, data


def _status_map(data: Dict[str, Any]) -> Dict[str, str]:
    return {c["id"]: c["status"] for c in data["checks"]}


def _result_for(data: Dict[str, Any], cid: str) -> Dict[str, Any]:
    for c in data["checks"]:
        if c["id"] == cid:
            return c
    raise KeyError(cid)


# ---------------------------------------------------------------------------
# good page → deliverable
# ---------------------------------------------------------------------------


def test_good_page_all_pass_deliverable(tmp_path: Path) -> None:
    html = _good_html(sticky=False)
    # no video, no oversize, no related self-links, short list
    html_path, cfg, site = _seed(tmp_path, html)
    code, data = _run_json(html_path, cfg, asset_root=site, base_url="https://example.com/page.html")
    assert code == pc.EXIT_OK
    assert data["deliverable"] is True
    assert data["blocking_fail"] == []
    sm = _status_map(data)
    assert sm["soft_404"] == "pass"
    assert sm["links_internal"] == "pass"
    assert sm["language"] == "pass"
    assert sm["file_size"] == "pass"
    assert sm["pagination"] == "pass"
    assert sm["video_attrs"] == "pass"
    assert sm["related_no_self"] == "pass"
    # sticky with no candidates still manual (never auto-pass)
    assert sm["sticky_cover"] == "manual"
    t = data["total"]
    assert t["count"] == 8
    assert t["pass"] + t["fail"] + t["skip"] + t["manual"] == t["count"]
    assert t["pass"] >= 7
    assert t["manual"] == 1


# ---------------------------------------------------------------------------
# soft_404
# ---------------------------------------------------------------------------


def test_soft_404_phrase_fail(tmp_path: Path) -> None:
    html = _good_html(main_extra="<p>Error 404 page missing</p>")
    html_path, cfg, site = _seed(tmp_path, html)
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert code == pc.EXIT_FAIL
    assert _status_map(data)["soft_404"] == "fail"
    assert "soft_404" in data["blocking_fail"]
    assert data["deliverable"] is False


def test_soft_404_too_short_fail(tmp_path: Path) -> None:
    html = """<!DOCTYPE html><html lang="th"><body><main>สั้น</main></body></html>"""
    html_path, cfg, site = _seed(tmp_path, html)
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert code == pc.EXIT_FAIL
    r = _result_for(data, "soft_404")
    assert r["status"] == "fail"
    assert "too short" in r["reason"]


def test_soft_404_main_missing_fail(tmp_path: Path) -> None:
    # no main, no article, no body → fail closed
    html = """<!DOCTYPE html><html lang="th"><div>orphan content without body tag properly... wait we need no body
    """
    # Actually HTMLParser will not create body. Use empty document-ish
    html = """<!DOCTYPE html><html lang="th"></html>"""
    html_path, cfg, site = _seed(tmp_path, html)
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert code == pc.EXIT_FAIL
    r = _result_for(data, "soft_404")
    assert r["status"] == "fail"
    assert "not found" in r["reason"].lower() or "too short" in r["reason"]


# ---------------------------------------------------------------------------
# links_internal
# ---------------------------------------------------------------------------


def test_links_internal_missing_file_fail(tmp_path: Path) -> None:
    html = _good_html(
        main_extra='<a href="/missing-page.html">gone</a>',
    )
    html_path, cfg, site = _seed(tmp_path, html)
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert code == pc.EXIT_FAIL
    r = _result_for(data, "links_internal")
    assert r["status"] == "fail"
    assert "missing" in r["reason"]


def test_links_internal_existing_pass(tmp_path: Path) -> None:
    html = _good_html(main_extra='<a href="./ok.html">ok</a>')
    html_path, cfg, site = _seed(tmp_path, html)
    _write(site / "ok.html", "<html lang='th'><body>ok</body></html>")
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert _status_map(data)["links_internal"] == "pass"
    # may still fail other checks? soft_404 etc should pass
    assert "links_internal" not in data["blocking_fail"]


def test_links_internal_external_skipped(tmp_path: Path) -> None:
    html = _good_html(
        main_extra=(
            '<a href="https://other.example.com/x.html">ext</a>'
            '<a href="mailto:a@b.com">mail</a>'
            '<a href="tel:+66123">tel</a>'
            '<a href="#section">frag</a>'
        ),
    )
    html_path, cfg, site = _seed(tmp_path, html)
    code, data = _run_json(
        html_path, cfg, asset_root=site, base_url="https://example.com/page.html"
    )
    r = _result_for(data, "links_internal")
    assert r["status"] == "pass"
    assert r["detail"]["skipped"] >= 4


# ---------------------------------------------------------------------------
# language
# ---------------------------------------------------------------------------


def test_language_missing_lang_fail(tmp_path: Path) -> None:
    html = f"""<!DOCTYPE html><html><body><main><p>{_MAIN_BODY}</p></main></body></html>"""
    html_path, cfg, site = _seed(tmp_path, html)
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert code == pc.EXIT_FAIL
    r = _result_for(data, "language")
    assert r["status"] == "fail"
    assert "missing" in r["reason"].lower()


def test_language_wrong_lang_fail(tmp_path: Path) -> None:
    html = _good_html(lang="fr")
    html_path, cfg, site = _seed(tmp_path, html)
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert code == pc.EXIT_FAIL
    r = _result_for(data, "language")
    assert r["status"] == "fail"
    assert "not in allowed" in r["reason"]


def test_language_bilingual_missing_en_fail(tmp_path: Path) -> None:
    cfg_text = GOOD_CONFIG.replace("bilingual: false", "bilingual: true")
    html = _good_html(main_extra="<p>ข่าววันนี้</p>")  # TH only, no EN marker
    html_path, cfg, site = _seed(tmp_path, html, cfg_text)
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert code == pc.EXIT_FAIL
    r = _result_for(data, "language")
    assert r["status"] == "fail"
    assert "EN" in r["reason"] or "bilingual" in r["reason"].lower()


def test_language_bilingual_pass(tmp_path: Path) -> None:
    cfg_text = GOOD_CONFIG.replace("bilingual: false", "bilingual: true")
    html = _good_html(main_extra="<p>ข่าว News today</p>")
    html_path, cfg, site = _seed(tmp_path, html, cfg_text)
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert _status_map(data)["language"] == "pass"


# ---------------------------------------------------------------------------
# file_size
# ---------------------------------------------------------------------------


def test_file_size_hero_over_cap_fail(tmp_path: Path) -> None:
    # 400 KB hero > 300
    html = _good_html(
        main_extra='<div class="hero"><img src="./hero.jpg" alt="h"/></div>',
    )
    html_path, cfg, site = _seed(
        tmp_path, html, assets={"hero.jpg": 400 * 1024}
    )
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert code == pc.EXIT_FAIL
    r = _result_for(data, "file_size")
    assert r["status"] == "fail"
    assert "400" in r["reason"] or "hero" in r["reason"].lower()
    # detail reports KB
    offenders = r["detail"]["offenders"]
    assert any(o.get("kb", 0) >= 399 for o in offenders)


def test_file_size_content_ok(tmp_path: Path) -> None:
    html = _good_html(main_extra='<img src="./pic.png" alt="p"/>')
    html_path, cfg, site = _seed(
        tmp_path, html, assets={"pic.png": 100 * 1024}
    )
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert _status_map(data)["file_size"] == "pass"


def test_file_size_video_over_cap_fail(tmp_path: Path) -> None:
    html = _good_html(
        main_extra='<video src="./big.mp4" muted playsinline></video>',
        head_extra="<style>@media (prefers-reduced-motion: reduce){}</style>",
    )
    html_path, cfg, site = _seed(
        tmp_path, html, assets={"big.mp4": 3 * 1024 * 1024}
    )
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert code == pc.EXIT_FAIL
    r = _result_for(data, "file_size")
    assert r["status"] == "fail"
    assert "video" in r["reason"].lower() or any(
        o.get("class") == "video" for o in r["detail"]["offenders"]
    )


def test_file_size_missing_file_fail(tmp_path: Path) -> None:
    html = _good_html(main_extra='<img src="./nope.png" alt="x"/>')
    html_path, cfg, site = _seed(tmp_path, html)
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert code == pc.EXIT_FAIL
    r = _result_for(data, "file_size")
    assert r["status"] == "fail"
    assert "missing" in r["reason"].lower()


def test_file_size_escape_path_fail(tmp_path: Path) -> None:
    html = _good_html(main_extra='<img src="../secret.png" alt="x"/>')
    html_path, cfg, site = _seed(tmp_path, html)
    # create file outside site so if escape weren't blocked it might exist
    _write_bytes(tmp_path / "secret.png", 10)
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert code == pc.EXIT_FAIL
    r = _result_for(data, "file_size")
    assert r["status"] == "fail"
    assert "escape" in r["reason"].lower() or any(
        "escape" in str(o.get("reason", "")).lower() for o in r["detail"]["offenders"]
    )


# ---------------------------------------------------------------------------
# pagination
# ---------------------------------------------------------------------------


def _list_html(n: int, with_control: bool) -> str:
    items = "".join(f"<li class='card'>item {i}</li>" for i in range(n))
    control = '<nav class="pagination" aria-label="pagination">1 2 3</nav>' if with_control else ""
    return _good_html(
        main_extra=f'<ul class="list" data-list="1">{items}</ul>{control}',
    )


def test_pagination_many_items_no_control_fail(tmp_path: Path) -> None:
    html_path, cfg, site = _seed(tmp_path, _list_html(20, False))
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert code == pc.EXIT_FAIL
    assert _status_map(data)["pagination"] == "fail"


def test_pagination_many_items_with_control_pass(tmp_path: Path) -> None:
    html_path, cfg, site = _seed(tmp_path, _list_html(20, True))
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert _status_map(data)["pagination"] == "pass"


def test_pagination_few_items_no_control_pass(tmp_path: Path) -> None:
    html_path, cfg, site = _seed(tmp_path, _list_html(5, False))
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert _status_map(data)["pagination"] == "pass"


# ---------------------------------------------------------------------------
# video_attrs
# ---------------------------------------------------------------------------


def test_video_autoplay_without_muted_fail(tmp_path: Path) -> None:
    html = _good_html(
        main_extra='<video autoplay playsinline src="./v.mp4"></video>',
        head_extra="<style>@media (prefers-reduced-motion: reduce){}</style>",
    )
    html_path, cfg, site = _seed(tmp_path, html, assets={"v.mp4": 100})
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert code == pc.EXIT_FAIL
    r = _result_for(data, "video_attrs")
    assert r["status"] == "fail"
    assert "muted" in r["reason"].lower()


def test_video_autoplay_muted_playsinline_ok(tmp_path: Path) -> None:
    html = _good_html(
        main_extra='<video autoplay muted playsinline src="./v.mp4"></video>',
        head_extra="<style>@media (prefers-reduced-motion: reduce){video{display:none}}</style>",
    )
    html_path, cfg, site = _seed(tmp_path, html, assets={"v.mp4": 100})
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert _status_map(data)["video_attrs"] == "pass"


def test_video_autoplay_no_reduced_motion_fail(tmp_path: Path) -> None:
    html = _good_html(
        main_extra='<video autoplay muted playsinline src="./v.mp4"></video>',
        # no prefers-reduced-motion anywhere
    )
    html_path, cfg, site = _seed(tmp_path, html, assets={"v.mp4": 100})
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert code == pc.EXIT_FAIL
    r = _result_for(data, "video_attrs")
    assert r["status"] == "fail"
    assert "prefers-reduced-motion" in r["reason"]


# ---------------------------------------------------------------------------
# related_no_self
# ---------------------------------------------------------------------------


def test_related_self_link_fail(tmp_path: Path) -> None:
    html = _good_html(
        body_extra=(
            '<aside class="related" data-related="1">'
            '<a href="https://example.com/page.html">self</a>'
            '<a href="/other.html">other</a>'
            "</aside>"
        ),
    )
    html_path, cfg, site = _seed(tmp_path, html)
    _write(site / "other.html", "x")
    code, data = _run_json(
        html_path, cfg, asset_root=site, base_url="https://example.com/page.html"
    )
    assert code == pc.EXIT_FAIL
    r = _result_for(data, "related_no_self")
    assert r["status"] == "fail"
    assert "self" in r["reason"].lower() or "current page" in r["reason"].lower()


def test_related_no_self_ok(tmp_path: Path) -> None:
    html = _good_html(
        body_extra=(
            '<aside class="related">'
            '<a href="/other.html">other</a>'
            "</aside>"
        ),
    )
    html_path, cfg, site = _seed(tmp_path, html)
    _write(site / "other.html", "x")
    code, data = _run_json(
        html_path, cfg, asset_root=site, base_url="https://example.com/page.html"
    )
    assert _status_map(data)["related_no_self"] == "pass"


# ---------------------------------------------------------------------------
# sticky_cover — manual never pass/fail
# ---------------------------------------------------------------------------


def test_sticky_cover_manual_not_blocking(tmp_path: Path) -> None:
    html = _good_html(sticky=True)
    html_path, cfg, site = _seed(tmp_path, html)
    code, data = _run_json(html_path, cfg, asset_root=site)
    r = _result_for(data, "sticky_cover")
    assert r["status"] == "manual"
    assert r["status"] not in ("pass", "fail")
    assert "sticky" in r["reason"].lower() or r["detail"]["candidates"]
    # exit 0 if no other blocking fails
    assert code == pc.EXIT_OK
    assert data["deliverable"] is True
    assert "sticky_cover" in data["manual_pending"]
    assert "sticky_cover" not in data["blocking_fail"]


# ---------------------------------------------------------------------------
# config / usage errors → exit 2
# ---------------------------------------------------------------------------


def test_config_missing_exit_2(tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    html_path = _write(site / "page.html", _good_html())
    code, text = _run(html_path, tmp_path / "nope.yaml")
    # config path doesn't exist
    assert code == pc.EXIT_ERR


def test_config_default_path_works(tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    html_path = _write(site / "page.html", _good_html())
    _write(site / ".work" / "page-check.yaml", GOOD_CONFIG)
    import io
    from contextlib import redirect_stdout, redirect_stderr

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = pc.main([str(html_path), "--asset-root", str(site), "--json"])
    assert code == pc.EXIT_OK
    data = json.loads(out.getvalue())
    assert data["deliverable"] is True


def test_unknown_check_id_exit_2(tmp_path: Path) -> None:
    bad = GOOD_CONFIG + "\n  totally_unknown: {blocking: true}\n"
    # insert unknown into checks — append after sticky line via replace
    bad = GOOD_CONFIG.replace(
        "sticky_cover: {blocking: false, manual: true}",
        "sticky_cover: {blocking: false, manual: true}\n  totally_unknown: {blocking: true}",
    )
    html_path, cfg, site = _seed(tmp_path, _good_html(), bad)
    code, text = _run(html_path, cfg, asset_root=site)
    assert code == pc.EXIT_ERR
    assert "unknown" in text.lower()


def test_html_missing_exit_2(tmp_path: Path) -> None:
    cfg = _write(tmp_path / "c.yaml", GOOD_CONFIG)
    import io
    from contextlib import redirect_stdout, redirect_stderr

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = pc.main([str(tmp_path / "no.html"), "--config", str(cfg)])
    assert code == pc.EXIT_ERR


# ---------------------------------------------------------------------------
# json shape + live counts
# ---------------------------------------------------------------------------


def test_json_shape_and_live_counts(tmp_path: Path) -> None:
    html_path, cfg, site = _seed(tmp_path, _good_html(sticky=True))
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert "file" in data
    assert "checks" in data
    assert "total" in data
    assert "blocking_fail" in data
    assert "deliverable" in data
    assert "manual_pending" in data
    for c in data["checks"]:
        assert set(c.keys()) >= {"id", "status", "blocking", "reason", "detail"}
        assert c["status"] in ("pass", "fail", "skip", "manual")
    t = data["total"]
    assert t["count"] == len(data["checks"])
    # live: sum of status buckets equals count
    assert t["pass"] + t["fail"] + t["skip"] + t["manual"] == t["count"]
    # recompute from checks — must match (no hardcoded)
    live = {"pass": 0, "fail": 0, "skip": 0, "manual": 0}
    for c in data["checks"]:
        live[c["status"]] += 1
    for k in live:
        assert t[k] == live[k]


def test_human_output_mentions_deliverable(tmp_path: Path) -> None:
    html_path, cfg, site = _seed(tmp_path, _good_html())
    code, text = _run(html_path, cfg, asset_root=site)
    assert code == pc.EXIT_OK
    assert "DELIVERABLE=YES" in text
    assert "soft_404: PASS" in text
    assert "sticky_cover: MANUAL" in text


def test_blocking_fail_exit_1_deliverable_no(tmp_path: Path) -> None:
    html = _good_html(lang="de")  # wrong language
    html_path, cfg, site = _seed(tmp_path, html)
    code, text = _run(html_path, cfg, asset_root=site)
    assert code == pc.EXIT_FAIL
    assert "DELIVERABLE=NO" in text


# ---------------------------------------------------------------------------
# unit: selector engine + mini yaml
# ---------------------------------------------------------------------------


def test_selector_engine_basic() -> None:
    root = pc.parse_html(
        '<div id="a" class="hero card" data-role="hero">'
        '<nav aria-label="pagination"></nav>'
        '<span class="x">t</span></div>'
    )
    assert pc.query_all(root, ".hero")
    assert pc.query_all(root, "#a")
    assert pc.query_all(root, "[data-role=hero]")
    assert pc.query_all(root, "nav[aria-label*=pag]")
    assert pc.query_all(root, ".hero, .nope")
    assert not pc.query_all(root, ".missing")


def test_visible_text_strips_script_style() -> None:
    root = pc.parse_html(
        "<div>hello<style>.x{color:red}</style><script>var a=1</script> world</div>"
    )
    div = pc.query_first(root, "div")
    assert div is not None
    text = pc.visible_text(div)
    assert "hello" in text and "world" in text
    assert "color" not in text
    assert "var a" not in text


def test_mini_yaml_loads_config() -> None:
    data = pc.load_yaml_text(GOOD_CONFIG, force_mini=True)
    assert isinstance(data, dict)
    assert "checks" in data
    assert data["soft404"]["min_content_chars"] == 50
    assert data["lang"]["allowed"] == ["th", "en"]
    assert data["checks"]["sticky_cover"]["manual"] is True


def test_validate_config_unknown() -> None:
    err = pc.validate_config({"checks": {"not_a_real_check": {"blocking": True}}})
    assert err is not None
    assert "unknown" in err.lower()


def test_fail_closed_empty_allowed_lang(tmp_path: Path) -> None:
    cfg = GOOD_CONFIG.replace('allowed: ["th", "en"]', "allowed: []")
    html_path, cfg_path, site = _seed(tmp_path, _good_html(), cfg)
    code, data = _run_json(html_path, cfg_path, asset_root=site)
    assert code == pc.EXIT_FAIL
    assert _status_map(data)["language"] == "fail"


# ---------------------------------------------------------------------------
# FIX 3b · stale item_selector (container has children but 0 items) -> fail
# ---------------------------------------------------------------------------


def test_pagination_stale_item_selector_fail(tmp_path: Path) -> None:
    big = "".join("<span>x</span>" for _ in range(20))
    html = _good_html(main_extra=f'<div class="list">{big}</div>')
    html_path, cfg, site = _seed(tmp_path, html)
    code, data = _run_json(html_path, cfg, asset_root=site)
    r = _result_for(data, "pagination")
    assert r["status"] == "fail", r
    assert "stale" in r["reason"]
    assert data["deliverable"] is False


def test_pagination_genuinely_empty_list_pass(tmp_path: Path) -> None:
    html = _good_html(main_extra='<div class="list"></div>')
    html_path, cfg, site = _seed(tmp_path, html)
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert _result_for(data, "pagination")["status"] == "pass"


# ---------------------------------------------------------------------------
# FIX 6 · reduced-motion must come from real CSS, not an HTML comment
# ---------------------------------------------------------------------------


def test_video_reduced_motion_comment_only_fail(tmp_path: Path) -> None:
    html = _good_html(
        head_extra="<!-- prefers-reduced-motion mentioned but not real CSS -->",
        main_extra='<video autoplay muted playsinline><source src="v.mp4"></video>',
    )
    html_path, cfg, site = _seed(tmp_path, html, assets={"v.mp4": 1024})
    code, data = _run_json(html_path, cfg, asset_root=site)
    r = _result_for(data, "video_attrs")
    assert r["status"] == "fail", r
    assert "reduced-motion" in r["reason"]


def test_video_reduced_motion_real_style_pass(tmp_path: Path) -> None:
    html = _good_html(
        head_extra=(
            "<style>@media (prefers-reduced-motion: reduce){*{animation:none}}</style>"
        ),
        main_extra='<video autoplay muted playsinline><source src="v.mp4"></video>',
    )
    html_path, cfg, site = _seed(tmp_path, html, assets={"v.mp4": 1024})
    code, data = _run_json(html_path, cfg, asset_root=site)
    assert _result_for(data, "video_attrs")["status"] == "pass"
