"""Register the work ledger CLI without changing the model's core tool surface."""
from hermes_cli.task_delivery.cli import configure


def build_work_parser(subparsers):
    return configure(subparsers.add_parser('work', help='Track task ownership, evidence, review and acceptance'))
