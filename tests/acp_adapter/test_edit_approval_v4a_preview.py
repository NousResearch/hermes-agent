"""A single-file V4A edit card shows the file before and after the patch, not the raw patch text.

ACP clients render ``oldText``/``newText`` as a diff of the file. Showing the patch itself as
``newText`` made every V4A edit read as replacing the file with the patch, and a Delete File read
as creating a file whose body is the patch.
"""

import json

import pytest

from acp_adapter.edit_approval import build_edit_proposal, set_edit_approval_requester
from model_tools import handle_function_call


@pytest.fixture
def recorded(tmp_path):
    """``(workdir, proposals)``: ``a.txt`` holds two lines; every edit is approved and recorded."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    proposals = []

    def requester(proposal):
        proposals.append(proposal)
        return True

    set_edit_approval_requester(requester)
    yield tmp_path, proposals
    set_edit_approval_requester(None)


@pytest.mark.parametrize("header, body, name, before, after", [
    ("Update", "@@\n-beta\n+delta\n", "a.txt", "alpha\nbeta\n", "alpha\ndelta\n"),
    ("Add", "+added\n+second\n", "new.txt", None, "added\nsecond"),  # V4A Add writes no final newline
    ("Delete", "", "a.txt", "alpha\nbeta\n", ""),
])
def test_v4a_card_is_the_file_before_and_after_the_patch(recorded, header, body, name, before, after):
    workdir, proposals = recorded
    target = workdir / name
    patch = f"*** Begin Patch\n*** {header} File: {target}\n{body}*** End Patch"

    result = json.loads(handle_function_call("patch", {"mode": "patch", "patch": patch}))

    assert "error" not in result, result
    [proposal] = proposals
    assert (proposal.old_text, proposal.new_text) == (before, after)
    # The card matches what the tool then did to the file.
    assert (target.read_text(encoding="utf-8") if target.exists() else "") == after


def test_v4a_patch_that_would_not_apply_shows_the_exact_patch(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    patch = f"*** Begin Patch\n*** Update File: {tmp_path / 'a.txt'}\n@@\n-missing\n+delta\n*** End Patch"

    proposal = build_edit_proposal("patch", {"mode": "patch", "patch": patch})

    assert (proposal.old_text, proposal.new_text) == ("alpha\nbeta\n", patch)
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_v4a_preview_never_touches_the_file(tmp_path):
    """The patch runs against an in-memory copy: building the card changes nothing on disk."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    for header, body in (("Update", "@@\n-beta\n+delta\n"), ("Delete", "")):
        patch = f"*** Begin Patch\n*** {header} File: {tmp_path / 'a.txt'}\n{body}*** End Patch"
        build_edit_proposal("patch", {"mode": "patch", "patch": patch})
    build_edit_proposal("patch", {"mode": "patch",
                                  "patch": f"*** Begin Patch\n*** Add File: {tmp_path / 'b.txt'}\n+x\n*** End Patch"})

    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert not (tmp_path / "b.txt").exists()
