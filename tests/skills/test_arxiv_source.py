from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "research"
    / "arxiv"
    / "scripts"
    / "fetch_arxiv_source.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("arxiv_source", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return load_module()


def make_tar_members(
    files: list[tuple[str, bytes]], links: dict[str, str] | None = None
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, data in files:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        for name, target in (links or {}).items():
            member = tarfile.TarInfo(name)
            member.type = tarfile.SYMTYPE
            member.linkname = target
            archive.addfile(member)
    return output.getvalue()


def make_tar(files: dict[str, bytes], links: dict[str, str] | None = None) -> bytes:
    return make_tar_members(list(files.items()), links=links)


def download(
    mod, data: bytes, url: str, content_type: str = "application/octet-stream"
):
    return mod.Download(data=data, content_type=content_type, final_url=url)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2402.03300", "2402.03300"),
        ("arXiv:2402.03300v3", "2402.03300v3"),
        ("https://arxiv.org/pdf/2402.03300v2.pdf?download=1", "2402.03300v2"),
        ("https://export.arxiv.org/abs/hep-th/0601001", "hep-th/0601001"),
    ],
)
def test_normalize_arxiv_id(mod, value, expected):
    assert mod.normalize_arxiv_id(value) == expected


@pytest.mark.parametrize(
    "value",
    ["../etc/passwd", "https://example.com/abs/2402.03300", "arxiv.org/foo"],
)
def test_normalize_arxiv_id_rejects_unsafe_or_invalid_values(mod, value):
    with pytest.raises(ValueError):
        mod.normalize_arxiv_id(value)


def test_source_archive_selects_main_expands_includes_and_summarizes_bib(mod):
    archive = make_tar({
        "paper/main.tex": b"""\\documentclass{article}
\\begin{document}
\\input{sections/intro}
% \\input{sections/ignored}
\\bibliography{refs}
\\end{document}
""",
        "paper/sections/intro.tex": b"Introduction.\\input{details}\n",
        "paper/sections/details.tex": b"Detailed result.\n",
        "paper/sections/ignored.tex": b"SHOULD NOT APPEAR\n",
        "paper/supplement.tex": b"Supplement only\n",
        "paper/refs.bib": b"""@article{smith2025,
  title = {A {Useful} Result},
  author = {Smith, Jane and Doe, John},
  year = {2025}
}
""",
    })

    rendered = mod.render_latex_source(archive)

    assert rendered.startswith("% arXiv main document: paper/main.tex")
    assert "Introduction." in rendered
    assert "Detailed result." in rendered
    assert "SHOULD NOT APPEAR" not in rendered
    assert "begin included: paper/sections/intro.tex" in rendered
    assert (
        "% - smith2025: A Useful Result | Smith, Jane and Doe, John | 2025" in rendered
    )


def test_source_archive_marks_include_cycles(mod):
    archive = make_tar({
        "main.tex": b"\\documentclass{article}\\input{a}",
        "a.tex": b"A\\input{main}",
    })

    rendered = mod.render_latex_source(archive)

    assert "cyclic include omitted: main.tex -> a.tex -> main.tex" in rendered


def test_source_archive_rejects_duplicate_non_text_members(mod):
    archive = make_tar_members([
        ("main.tex", b"\\documentclass{article}\\begin{document}OK"),
        ("figures/result.png", b"first"),
        ("figures/result.png", b"second"),
    ])

    with pytest.raises(mod.SourceFormatError, match="duplicate archive member"):
        mod.render_latex_source(archive)


@pytest.mark.parametrize(
    "archive",
    [
        make_tar({"../escape.tex": b"\\documentclass{article}"}),
        make_tar(
            {"main.tex": b"\\documentclass{article}"},
            links={"linked.tex": "main.tex"},
        ),
    ],
)
def test_source_archive_rejects_traversal_and_links(mod, archive):
    with pytest.raises(mod.SourceFormatError):
        mod.render_latex_source(archive)


def test_fetch_paper_prefers_source_and_writes_tex(mod, tmp_path):
    archive = make_tar({"main.tex": b"\\documentclass{article}\\begin{document}OK"})
    calls = []

    def fetcher(url):
        calls.append(url)
        return download(mod, archive, url)

    result = mod.fetch_paper("2402.03300", tmp_path / "paper", fetcher)

    assert result.kind == "latex"
    assert result.path == tmp_path / "paper.tex"
    assert result.path.read_text().endswith("OK")
    assert calls == ["https://arxiv.org/src/2402.03300"]


def test_fetch_paper_falls_back_to_clean_html(mod, tmp_path):
    calls = []

    def fetcher(url):
        calls.append(url)
        if "/src/" in url:
            raise mod.FetchError("source unavailable")
        html = b"""<html><head><meta charset="utf-8"><title>Noise</title></head><body>
<nav>Menu</nav><main><h1>Paper title</h1><p>Main result.</p>
<script>ignore()</script></main><footer>Footer</footer></body></html>"""
        return download(mod, html, url, "text/html")

    result = mod.fetch_paper("2402.03300", tmp_path / "paper.tex", fetcher)

    assert result.kind == "html"
    assert result.path == tmp_path / "paper.txt"
    assert result.path.read_text() == "Paper title\nMain result.\n"
    assert calls == [
        "https://arxiv.org/src/2402.03300",
        "https://arxiv.org/html/2402.03300",
    ]


def test_fetch_paper_falls_back_to_pdf(mod, tmp_path):
    calls = []

    def fetcher(url):
        calls.append(url)
        if "/pdf/" not in url:
            raise mod.FetchError("unavailable")
        return download(mod, b"%PDF-1.7\nfixture", url, "application/pdf")

    result = mod.fetch_paper("hep-th/0601001", tmp_path / "paper", fetcher)

    assert result.kind == "pdf"
    assert result.path.read_bytes() == b"%PDF-1.7\nfixture"
    assert calls[-1] == "https://arxiv.org/pdf/hep-th/0601001"


def test_fetch_paper_falls_back_after_unsafe_source_archive(mod, tmp_path):
    unsafe = make_tar({"../../escape.tex": b"\\documentclass{article}"})

    def fetcher(url):
        if "/src/" in url:
            return download(mod, unsafe, url)
        if "/html/" in url:
            return download(mod, b"<body><article>Safe fallback</article></body>", url)
        raise AssertionError("PDF should not be requested")

    result = mod.fetch_paper("2402.03300", tmp_path / "paper", fetcher)

    assert result.kind == "html"
    assert result.path.read_text() == "Safe fallback\n"


def test_fetch_paper_reports_all_failed_representations(mod, tmp_path):
    def fetcher(url):
        raise mod.FetchError(f"unavailable: {url}")

    with pytest.raises(mod.FetchError, match="source:.*html:.*pdf:"):
        mod.fetch_paper("2402.03300", tmp_path / "paper", fetcher)
