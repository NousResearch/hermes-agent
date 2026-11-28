import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/office_text_replace.py'
SPEC = importlib.util.spec_from_file_location('office_text_replace', SCRIPT)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
sys.path.insert(0, str(SCRIPT.parent))
COMMON_SPEC = importlib.util.spec_from_file_location(
    'docx_common_pure', SCRIPT.parent / 'docx_common.py',
)
docx_common = importlib.util.module_from_spec(COMMON_SPEC)
COMMON_SPEC.loader.exec_module(docx_common)
class Run:
    def __init__(self, text, element=None):
        self.text = text
        self._r = element


class OfficeTextReplaceTests(unittest.TestCase):
    def replace(self, values, old, new):
        runs = [Run(value) for value in values]
        count = helper.replace_in_runs(runs, old, new)
        return count, [run.text for run in runs]

    def test_replacement_containing_needle_is_not_reprocessed(self):
        self.assertEqual(self.replace(['cat'], 'cat', 'cats'), (1, ['cats']))

    def test_identity_replacement_terminates(self):
        self.assertEqual(self.replace(['cat'], 'cat', 'cat'), (1, ['cat']))

    def test_whole_run_and_cross_run_matches_are_both_replaced(self):
        self.assertEqual(
            self.replace(['old', 'o', 'ld'], 'old', 'new'),
            (2, ['new', 'new', '']),
        )

    def test_multiple_matches_in_one_run(self):
        self.assertEqual(
            self.replace(['cat cat'], 'cat', 'x'),
            (2, ['x x']),
        )

    def test_soft_break_splits_searchable_run_groups(self):
        class Element:
            def __init__(self):
                self.next = None

            def getnext(self):
                return self.next

        first, second = Element(), Element()
        runs = [Run('old', first), Run('name', second)]
        groups = helper.contiguous_run_groups(runs)
        self.assertEqual(len(groups), 2)
        count = sum(helper.replace_in_runs(group, 'oldname', 'new')
                    for group in groups)
        self.assertEqual(count, 0)

    def test_linked_header_part_is_visited_once(self):
        class Container:
            def __init__(self, paragraphs=()):
                self.paragraphs = list(paragraphs)
                self.tables = []
                self._element = object()

        shared = Container(['shared'])
        empty = Container()
        section = type('Section', (), {
            'header': shared, 'footer': empty,
            'first_page_header': empty, 'first_page_footer': empty,
            'even_page_header': empty, 'even_page_footer': empty,
        })()
        document = Container(['body'])
        document.sections = [section, section]
        self.assertEqual(
            list(docx_common.iter_all_paragraphs(document)),
            ['body', 'shared'],
        )

    def test_merged_table_cell_is_visited_once(self):
        class Cell:
            def __init__(self, identity, paragraphs):
                self._tc = identity
                self.paragraphs = paragraphs
                self.tables = []

        identity = object()
        left = Cell(identity, ['merged'])
        right = Cell(identity, ['merged'])
        row = type('Row', (), {'cells': [left, right]})()
        table = type('Table', (), {'rows': [row]})()
        self.assertEqual(list(docx_common._iter_table(table)), ['merged'])

    @unittest.skipUnless(importlib.util.find_spec('docx'),
                         'python-docx is not installed')
    def test_replacement_includes_hyperlink_runs(self):
        from docx import Document
        from docx.oxml import OxmlElement

        paragraph = Document().add_paragraph('Visit ')
        hyperlink = OxmlElement('w:hyperlink')
        run = OxmlElement('w:r')
        text = OxmlElement('w:t')
        text.text = 'old site'
        run.append(text)
        hyperlink.append(run)
        paragraph._p.append(hyperlink)

        self.assertEqual(
            docx_common.replace_in_paragraph(paragraph, 'old', 'new'), 1)
        self.assertEqual(
            [run.text for run in docx_common.paragraph_runs(paragraph)],
            ['Visit ', 'new site'],
        )

if __name__ == '__main__':
    unittest.main()
