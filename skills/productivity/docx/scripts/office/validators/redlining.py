"""
Validator for tracked changes in Word documents.

Detects untracked edits in word/document.xml: text that differs from the
original without a <w:ins>/<w:del> wrapper recording it. The tracked changes
that are new relative to the original are undone, and the result is compared
against the original; whatever text still differs was edited without being
tracked.

Only the document body is compared. Headers, footers, footnotes and endnotes
are separate parts and are not checked.
"""

import subprocess
import tempfile
import zipfile
from pathlib import Path

import defusedxml.ElementTree as ET
from defusedxml.common import DefusedXmlException

from helpers import rendered_text, safe_extract


class RedliningValidator:

    def __init__(self, unpacked_dir, original_docx, verbose=False):
        self.unpacked_dir = Path(unpacked_dir)
        self.original_docx = Path(original_docx)
        self.verbose = verbose
        self.namespaces = {
            "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        }

    def repair(self) -> int:
        return 0

    def validate(self):
        modified_file = self.unpacked_dir / "word" / "document.xml"
        if not modified_file.exists():
            print(f"FAILED - Modified document.xml not found at {modified_file}")
            return False

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                with zipfile.ZipFile(self.original_docx, "r") as zip_ref:
                    safe_extract(zip_ref, temp_path)
            except Exception as e:
                print(f"FAILED - Error unpacking original docx: {e}")
                return False

            original_file = temp_path / "word" / "document.xml"
            if not original_file.exists():
                print(
                    f"FAILED - Original document.xml not found in {self.original_docx}"
                )
                return False

            try:
                modified_tree = ET.parse(modified_file)
                modified_root = modified_tree.getroot()
                original_tree = ET.parse(original_file)
                original_root = original_tree.getroot()
            except (ET.ParseError, DefusedXmlException) as e:
                print(f"FAILED - Error parsing XML files: {e}")
                return False

            new_changes = self._new_tracked_changes(original_root, modified_root)
            self._remove_tracked_changes(modified_root, new_changes)

            modified_text = self._extract_text_content(modified_root)
            original_text = self._extract_text_content(original_root)

            if modified_text != original_text:
                error_message = self._generate_detailed_diff(
                    original_text, modified_text
                )
                print(error_message)
                return False

            if self.verbose:
                print(
                    f"PASSED - All {len(new_changes)} change(s) against the original "
                    "are properly tracked"
                )
            return True

    def _tracked_change_elements(self, root):
        ins_tag = f"{{{self.namespaces['w']}}}ins"
        del_tag = f"{{{self.namespaces['w']}}}del"
        return [elem for elem in root.iter() if elem.tag in (ins_tag, del_tag)]

    def _rendered_text(self, elem):
        preserve = elem.get("{http://www.w3.org/XML/1998/namespace}space") == "preserve"
        return rendered_text(elem.text or "", preserve)

    def _text_elements(self, elem):
        w = self.namespaces["w"]
        return [
            node
            for node in elem.iter()
            if node.tag in (f"{{{w}}}t", f"{{{w}}}delText")
        ]

    def _tracked_change_key(self, elem):
        w = self.namespaces["w"]
        text = "".join(self._rendered_text(node) for node in self._text_elements(elem))
        return (elem.tag, elem.get(f"{{{w}}}author"), elem.get(f"{{{w}}}date"), text)

    def _new_tracked_changes(self, original_root, modified_root):
        original = self._tracked_change_elements(original_root)
        modified = self._tracked_change_elements(modified_root)

        pool = {}
        for elem in original:
            pool.setdefault(self._tracked_change_key(elem), []).append(elem)

        matched, leftover = set(), []
        for elem in modified:
            bucket = pool.get(self._tracked_change_key(elem))
            if bucket:
                matched.add(bucket.pop())
            else:
                leftover.append(elem)

        def group(elem):
            return self._tracked_change_key(elem)[:3]

        def text_of(elems):
            return "".join(self._tracked_change_key(e)[3] for e in elems)

        unmatched_original = {}
        for elem in original:
            if elem not in matched:
                unmatched_original.setdefault(group(elem), []).append(elem)

        by_group = {}
        for elem in leftover:
            by_group.setdefault(group(elem), []).append(elem)

        new = set()
        for key, elems in by_group.items():
            rebuilt = text_of(elems)
            if rebuilt and rebuilt == text_of(unmatched_original.get(key, [])):
                continue  
            new.update(elems)
        return new

    def _generate_detailed_diff(self, original_text, modified_text):
        error_parts = [
            "FAILED - Document text doesn't match after removing the tracked changes",
            "",
            "Likely causes:",
            "  1. Modified text inside another author's <w:ins> or <w:del> tags",
            "  2. Made edits without proper tracked changes",
            "  3. Didn't nest <w:del> inside <w:ins> when deleting another's insertion",
            "  4. Rewrote another author's <w:ins>/<w:del> and changed its text on",
            "     the way. A tracked change from the original is recognised by its",
            "     author, date and text; anything that doesn't reproduce one exactly",
            "     reads as new, and the text it carried is reported missing.",
            "",
            "For pre-redlined documents, use correct patterns:",
            "  - To reject another's INSERTION: Nest <w:del> inside their <w:ins>",
            "  - To reject PART of one: nest <w:del> around only the runs you reject.",
            "    Their <w:ins> may be split around it, so long as the pieces keep",
            "    their author and date and still spell out the same text.",
            "  - To restore another's DELETION: Add new <w:ins> AFTER their <w:del>",
            "",
        ]

        git_diff = self._get_git_word_diff(original_text, modified_text)
        if git_diff:
            error_parts.extend(["Differences:", "============", git_diff])
        else:
            error_parts.append("Unable to generate word diff (git not available)")

        return "\n".join(error_parts)

    def _get_git_word_diff(self, original_text, modified_text):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                original_file = temp_path / "original.txt"
                modified_file = temp_path / "modified.txt"

                original_file.write_text(original_text, encoding="utf-8")
                modified_file.write_text(modified_text, encoding="utf-8")

                result = subprocess.run(
                    [
                        "git",
                        "diff",
                        "--word-diff=plain",
                        "--word-diff-regex=.",  
                        "-U0",  
                        "--no-index",
                        str(original_file),
                        str(modified_file),
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.stdout.strip():
                    lines = result.stdout.split("\n")
                    content_lines = []
                    in_content = False
                    for line in lines:
                        if line.startswith("@@"):
                            in_content = True
                            continue
                        if in_content and line.strip():
                            content_lines.append(line)

                    if content_lines:
                        return "\n".join(content_lines)

                result = subprocess.run(
                    [
                        "git",
                        "diff",
                        "--word-diff=plain",
                        "-U0",  
                        "--no-index",
                        str(original_file),
                        str(modified_file),
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.stdout.strip():
                    lines = result.stdout.split("\n")
                    content_lines = []
                    in_content = False
                    for line in lines:
                        if line.startswith("@@"):
                            in_content = True
                            continue
                        if in_content and line.strip():
                            content_lines.append(line)
                    return "\n".join(content_lines)

        except (subprocess.CalledProcessError, FileNotFoundError, Exception):
            pass

        return None

    def _remove_tracked_changes(self, root, targets):
        ins_tag = f"{{{self.namespaces['w']}}}ins"
        del_tag = f"{{{self.namespaces['w']}}}del"

        for parent in root.iter():
            to_remove = []
            for child in parent:
                if child.tag == ins_tag and child in targets:
                    to_remove.append(child)
            for elem in to_remove:
                parent.remove(elem)

        deltext_tag = f"{{{self.namespaces['w']}}}delText"
        t_tag = f"{{{self.namespaces['w']}}}t"

        for parent in root.iter():
            to_process = []
            for child in parent:
                if child.tag == del_tag and child in targets:
                    to_process.append((child, list(parent).index(child)))

            for del_elem, del_index in reversed(to_process):
                for elem in del_elem.iter():
                    if elem.tag == deltext_tag:
                        elem.tag = t_tag

                for child in reversed(list(del_elem)):
                    parent.insert(del_index, child)
                parent.remove(del_elem)

    def _extract_text_content(self, root):
        p_tag = f"{{{self.namespaces['w']}}}p"
        t_tag = f"{{{self.namespaces['w']}}}t"

        paragraphs = []
        for p_elem in root.findall(f".//{p_tag}"):
            text_parts = []
            for t_elem in p_elem.findall(f".//{t_tag}"):
                text_parts.append(self._rendered_text(t_elem))
            paragraph_text = "".join(text_parts)
            if paragraph_text:
                paragraphs.append(paragraph_text)

        return "\n".join(paragraphs)


if __name__ == "__main__":
    raise RuntimeError("This module should not be run directly.")
