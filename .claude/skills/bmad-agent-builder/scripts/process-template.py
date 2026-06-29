#!/usr/bin/env python3
"""Process BMad agent template files.

Performs deterministic variable substitution and conditional block processing
on template files from assets/. Replaces {varName} placeholders with provided
values and evaluates {if-X}...{/if-X} conditional blocks, keeping content
when the condition is in the --true list and removing the entire block otherwise.

Any {if-X} or {/if-X} marker still present after processing is a defect (a
malformed or mismatched block the emitted agent would ship verbatim): the
script exits 3 and names the markers. Remaining {token} placeholders are
reported in the --json metadata as tokens_remaining, not failed, because they
may be runtime-resolution tokens such as {project-root} or {agent.<name>} —
the builder judges that list against the build-time token set.
"""

# /// script
# requires-python = ">=3.9"
# ///

from __future__ import annotations

import argparse
import json
import re
import sys


def process_conditionals(text: str, true_conditions: set[str]) -> tuple[str, list[str], list[str]]:
    """Process {if-X}...{/if-X} conditional blocks, innermost first.

    Returns (processed_text, conditions_true, conditions_false).
    """
    conditions_true: list[str] = []
    conditions_false: list[str] = []

    # Process innermost blocks first to handle nesting
    pattern = re.compile(
        r'\{if-([a-zA-Z0-9_-]+)\}(.*?)\{/if-\1\}',
        re.DOTALL,
    )

    changed = True
    while changed:
        changed = False
        match = pattern.search(text)
        if match:
            changed = True
            condition = match.group(1)
            inner = match.group(2)

            if condition in true_conditions:
                # Keep the inner content, strip the markers
                # Remove a leading newline if the opening tag was on its own line
                replacement = inner
                if condition not in conditions_true:
                    conditions_true.append(condition)
            else:
                # Remove the entire block
                replacement = ''
                if condition not in conditions_false:
                    conditions_false.append(condition)

            text = text[:match.start()] + replacement + text[match.end():]

    # Clean up blank lines left by removed blocks: collapse 3+ consecutive
    # newlines down to 2 (one blank line)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text, conditions_true, conditions_false


def process_variables(text: str, variables: dict[str, str]) -> tuple[str, list[str]]:
    """Replace {varName} placeholders with provided values.

    Only replaces variables that are in the provided mapping.
    Leaves unmatched {variables} untouched (they may be runtime config).

    Returns (processed_text, list_of_substituted_var_names).
    """
    substituted: list[str] = []

    for name, value in variables.items():
        placeholder = '{' + name + '}'
        if placeholder in text:
            text = text.replace(placeholder, value)
            if name not in substituted:
                substituted.append(name)

    return text, substituted


def parse_var(s: str) -> tuple[str, str]:
    """Parse a key=value string. Raises argparse error on bad format."""
    if '=' not in s:
        raise argparse.ArgumentTypeError(
            f"Invalid variable format: '{s}' (expected key=value)"
        )
    key, _, value = s.partition('=')
    if not key:
        raise argparse.ArgumentTypeError(
            f"Invalid variable format: '{s}' (empty key)"
        )
    return key, value


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Process BMad agent template files with variable substitution and conditional blocks.',
    )
    parser.add_argument(
        'template',
        help='Path to the template file to process',
    )
    parser.add_argument(
        '-o', '--output',
        help='Write processed output to file (default: stdout)',
    )
    parser.add_argument(
        '--var',
        action='append',
        default=[],
        metavar='key=value',
        help='Variable substitution (repeatable). Example: --var skillName=my-agent',
    )
    parser.add_argument(
        '--true',
        action='append',
        default=[],
        dest='true_conditions',
        metavar='CONDITION',
        help='Condition name to treat as true (repeatable). Example: --true pulse --true evolvable',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        dest='json_output',
        help='Output processing metadata as JSON to stderr',
    )

    args = parser.parse_args()

    # Parse variables
    variables: dict[str, str] = {}
    for v in args.var:
        try:
            key, value = parse_var(v)
        except argparse.ArgumentTypeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
        variables[key] = value

    true_conditions = set(args.true_conditions)

    # Read template
    try:
        with open(args.template, encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Template file not found: {args.template}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"Error reading template: {e}", file=sys.stderr)
        return 1

    # Process: conditionals first, then variables
    content, conds_true, conds_false = process_conditionals(content, true_conditions)
    content, vars_substituted = process_variables(content, variables)

    # Leftover conditional markers mean a malformed/mismatched block that
    # would ship verbatim in the emitted agent.
    leftover_markers = sorted(set(re.findall(r'\{/?if-[a-zA-Z0-9_-]+\}', content)))
    if leftover_markers:
        print(
            f"Error: leftover conditional markers after processing: {', '.join(leftover_markers)}",
            file=sys.stderr,
        )
        return 3

    tokens_remaining = sorted(set(re.findall(r'\{[a-zA-Z][a-zA-Z0-9_.-]*\}', content)))

    # Write output
    output_file = args.output
    try:
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            sys.stdout.write(content)
    except OSError as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        return 1

    # JSON metadata to stderr
    if args.json_output:
        metadata = {
            'processed': True,
            'output_file': output_file or '<stdout>',
            'vars_substituted': vars_substituted,
            'conditions_true': conds_true,
            'conditions_false': conds_false,
            'tokens_remaining': tokens_remaining,
        }
        print(json.dumps(metadata, indent=2), file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
