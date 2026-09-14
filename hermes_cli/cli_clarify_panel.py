"""Bounded batch clarification panel and per-question editing drafts."""
from prompt_toolkit.utils import get_cwidth


def _clip(text, width):
    text = ' '.join(str(text).split())
    if get_cwidth(text) <= width:
        return text
    result = ''
    for char in text:
        if get_cwidth(result + char) > width - 1:
            break
        result += char
    return result + '…'


def render_batch(cli, state):
    from cli import _panel_box_width, _wrap_panel_text
    from hermes_cli.cli_tui_mixin import _Panel, _term_rows, _PANEL_RESERVED_BELOW

    questions = state['questions']
    active = state['active']
    entry = questions[active]
    answers = state['answers']
    reviewing = state.get('reviewing', False)
    width = _panel_box_width('Hermes needs your input', [entry['question']])
    inner = max(1, width - 2)
    available = max(5, _term_rows() - _PANEL_RESERVED_BELOW)
    panel = _Panel('class:clarify-border', width, 'Hermes needs your input', 'class:clarify-title')
    tabs = []
    for i, q in enumerate(questions):
        label = f"{i + 1}{'✓' if q['qid'] in answers else ''}"
        tabs.append(f'[{label}]' if i == active else label)
    tab_text = ' '.join(tabs) + f'  {len(answers)}/{len(questions)} answered'
    panel.row('class:clarify-title', _clip(tab_text, inner))
    budget = available - 4  # borders, tabs, action footer
    if reviewing:
        panel.row('class:clarify-selected', _clip('Submit all answers? Enter to submit.', inner))
        if budget > 1:
            panel.row('class:clarify-answer', _clip(f"{active + 1}: {answers.get(entry['qid'], '')}", inner))
        footer = 'Tab / Shift-Tab: review or edit · Ctrl+C: cancel'
    else:
        choices = state.get('choices') or []
        selected = state.get('selected', 0)
        checked = state.get('selected_indices') or set()
        multi = state.get('multi_select')
        freetext = cli._clarify_freetext
        # Reserve a visible choice even at small heights; only the active body is expanded.
        question_budget = max(1, min(3, budget - min(len(choices) + 1, 3)))
        question_rows = _wrap_panel_text(entry['question'], inner)
        if len(question_rows) > question_budget:
            question_rows = question_rows[:question_budget]
            question_rows[-1] = _clip(question_rows[-1] + ' …', inner)
        for row in question_rows:
            panel.row('class:clarify-question', row)
        remaining = budget - len(question_rows)
        if choices and remaining > 0:
            labels = choices + ['Other (type below)' if freetext else 'Other (type your answer)']
            focus = len(choices) if freetext else selected
            start = max(0, min(focus - remaining // 2, len(labels) - remaining))
            for index in range(start, min(len(labels), start + remaining)):
                cursor = '❯' if index == focus else ' '
                check = ('[x] ' if index in checked else '[ ] ') if multi else ''
                label = f'{cursor} {check}{index + 1}. {labels[index]}'
                if index == start and start > 0:
                    label += ' ↑'
                if index == start + remaining - 1 and index < len(labels) - 1:
                    label += ' ↓'
                style = 'class:clarify-selected' if index == focus else 'class:clarify-choice'
                panel.row(style, _clip(label, inner))
        elif remaining > 0:
            panel.row('class:clarify-active-other', _clip('Type your answer below.', inner))
        footer = 'Enter: save · Tab / Shift-Tab: switch'
        if multi:
            footer = 'Space: toggle · Enter: save · Tab: switch'
    panel.row('class:clarify-choice', _clip(footer, inner))
    return panel.close()


def save_draft(cli, state, buffer):
    if state.get('reviewing'):
        return
    state.setdefault('drafts', {})[state['active']] = {
        'selected': state['selected'],
        'selected_indices': set(state.get('selected_indices') or set()),
        'freetext': cli._clarify_freetext,
        'text': buffer.text if cli._clarify_freetext else '',
        'multi_base': cli._clarify_multi_base,
    }


def restore_draft(cli, state, index):
    draft = state.get('drafts', {}).get(index)
    if draft is None:
        # Open-ended saved answers must be editable without retyping.
        meta = state.get('answer_meta', {}).get(state['questions'][index]['qid'], {})
        cli._clarify_prefill = meta.get('other_text', '') if cli._clarify_freetext else ''
        return False
    state['selected'] = draft['selected']
    state['selected_indices'] = set(draft['selected_indices'])
    cli._clarify_freetext = draft['freetext']
    cli._clarify_multi_base = draft['multi_base']
    cli._clarify_prefill = draft['text']
    return True


def fill_composer(cli, buffer):
    buffer.text = cli._clarify_prefill if cli._clarify_freetext else ''
    buffer.cursor_position = len(buffer.text)
    cli._clarify_prefill = ''
