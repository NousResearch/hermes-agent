"""The canonical in-app links the LitKit tools hand Ana: the three target shapes, labels that
survive a markdown parser, targets that cannot end early, the 400-character cap, and the SOUL
paragraph that tells Ana to paste them as given."""

from __future__ import annotations

import re
from urllib.parse import unquote

import pytest
from markdown_it import MarkdownIt

from litco.litkit.links import LINK_MAX, document_link, file_link, folder_link

MATTER = "11111111-2222-3333-4444-555555555555"
LS = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
DOC = "00000000-0000-4000-8000-000000000001"
FILE = "00000000-0000-4000-8000-000000000002"
NAMES = ["2026-09-15 Exhibit A - Privilege Log.pdf", "Smith (final) [v2].pdf", "R&D, Q3 `draft` <x>.docx",
         "back\\slash.pdf"]


def parse(link: str) -> tuple[str, str]:
    """(label text, href) of the one link a markdown parser finds in ``link``, followed by prose."""
    tokens = MarkdownIt("commonmark").parseInline(link + " and then `code` and (more).")[0].children
    opens = [i for i, t in enumerate(tokens) if t.type == "link_open"]
    assert len(opens) == 1, link
    start = opens[0]
    end = next(i for i in range(start, len(tokens)) if tokens[i].type == "link_close")
    text = "".join(t.content for t in tokens[start + 1:end])
    after = "".join(t.content for t in tokens[end + 1:])
    assert after.startswith(" and then"), "the link swallowed the text after it"
    return text, tokens[start].attrs["href"]


def split(link: str) -> tuple[str, str]:
    match = re.fullmatch(r"\[(.*)\]\(([^()\s]*)\)", link, re.S)
    assert match, link
    return match.group(1), match.group(2)


def test_each_builder_has_its_shape():
    assert split(document_link(MATTER, DOC, "ABC0001"))[1] == f"/matters/{MATTER}?doc={DOC}"
    assert split(document_link(MATTER, DOC, "ABC0001", page=7))[1] == f"/matters/{MATTER}?doc={DOC}&page=7"
    assert split(file_link(LS, FILE, "a.pdf"))[1] == f"/litspace/matters/{LS}/documents/{FILE}"
    assert split(folder_link(LS, "Productions"))[1] == f"/litspace/matters/{LS}/files?path=Productions"


@pytest.mark.parametrize("page", [None, 0, -3, True, "2", 1.5])
def test_only_a_positive_int_page_is_kept(page):
    assert "page=" not in document_link(MATTER, DOC, "d", page=page)


@pytest.mark.parametrize("name", NAMES)
def test_label_round_trips_and_the_target_cannot_end_early(name):
    for link in (file_link(LS, FILE, name), document_link(MATTER, DOC, name)):
        text, href = parse(link)
        assert text == name
        assert href == split(link)[1]
        assert ")" not in href and "(" not in href and " " not in href


def test_folder_link_encodes_segments_and_keeps_separators():
    link = folder_link(LS, "Productions/Volume 1")
    text, href = parse(link)
    assert text == "Productions/Volume 1"
    path = href.split("?path=", 1)[1]
    assert path == "Productions/Volume%201"
    assert unquote(path) == "Productions/Volume 1"
    odd = folder_link(LS, "A (x)/B&C=D?/E")
    assert parse(odd)[1].split("?path=", 1)[1].count("/") == 2
    assert unquote(parse(odd)[1].split("?path=", 1)[1]) == "A (x)/B&C=D?/E"
    assert folder_link(LS, "Productions", "the productions").startswith("[the productions](")


@pytest.mark.parametrize("folder", ["", "  ", "/", None])
def test_the_vault_root_gives_no_folder_link(folder):
    assert folder_link(LS, folder) == ""


@pytest.mark.parametrize("bad", ["", "not-a-uuid", DOC + "x", "../" + DOC, None, 7])
def test_a_bad_id_raises(bad):
    with pytest.raises(ValueError):
        document_link(bad, DOC, "d")
    with pytest.raises(ValueError):
        document_link(MATTER, bad, "d")
    with pytest.raises(ValueError):
        file_link(bad, FILE, "f")
    with pytest.raises(ValueError):
        file_link(LS, bad, "f")
    with pytest.raises(ValueError):
        folder_link(bad, "Productions")


def test_line_breaks_become_spaces_and_long_labels_clip_at_160():
    assert parse(file_link(LS, FILE, "a\nb\r\nc"))[0] == "a b c"
    text = parse(file_link(LS, FILE, "x" * 170))[0]
    assert len(text) == 160 and text.endswith("…")


def test_no_link_exceeds_the_cap():
    target = split(file_link(LS, FILE, "f"))[1]
    long_name = "n" * 300
    link = file_link(LS, FILE, long_name)
    assert len(link) <= LINK_MAX and split(link)[1] == target
    deep = "/".join(["Folder name with spaces"] * 12)
    near = "/".join(["Folder name with spaces"] * 8)
    link = folder_link(LS, near)
    assert link and len(link) <= LINK_MAX
    text, href = parse(link)
    assert text.endswith("…") and near.startswith(text[:-1]) and unquote(href.split("?path=", 1)[1]) == near
    assert folder_link(LS, deep) == ""
    escapes = "[" * 300  # every label character doubles when escaped
    link = file_link(LS, FILE, escapes)
    text = parse(link)[0]
    assert len(link) <= LINK_MAX and text.endswith("…") and set(text[:-1]) == {"["}


def test_links_carry_no_origin():
    for link in (document_link(MATTER, DOC, "d", page=2), file_link(LS, FILE, "f"), folder_link(LS, "A/B")):
        assert "http" not in link and "//" not in link and split(link)[1].startswith("/")


def test_soul_renders_the_link_paragraph_and_both_placeholders(tmp_path):
    from tests.host.test_unit_and_init import env_for, init, rendered
    assert init.main(env_for(tmp_path)) == 0
    _config, soul = rendered(tmp_path)
    assert "{{" not in soul and "matter-42" in soul and "https://acme.litco.ai" in soul
    assert ("When a tool result gives a link for a document, a file, or a folder, use that link the first time "
            "you name the item in an answer, written exactly as the tool gave it.") in soul
    assert "never write a link that a tool did not give you" in soul
    assert "LitKit document id" not in soul
