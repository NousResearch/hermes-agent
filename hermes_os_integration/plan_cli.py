"""CLI entrypoint for compiling Hermes OS work graphs."""

import argparse
import json
import sys

from .work_graph import compile_work_graph, save_work_graph, serialize_work_graph
from .persistence import SQLiteRepository, persist_work_graph
from .scanners import scan_project
from .templates import TemplateCompiler, TemplateLoader
from .tasks import generate_tasks_from_work_graph, next_task_number_from_files, write_task_artifacts


PASS = 0
WARNING = 2
INVALID_REQUEST = 64


def build_parser():
    parser = argparse.ArgumentParser(prog="hermes plan")
    parser.add_argument("project")
    parser.add_argument("--projects-root")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--generate-tasks", action="store_true")
    parser.add_argument("--template", default="")
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--db", default="")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.project:
        return INVALID_REQUEST
    try:
        graph = _compile(args)
    except Exception as exc:
        return _emit_error(args, "plan_failed", str(exc))
    path = None
    if args.write and graph.nodes:
        project_path = graph.nodes[0].metadata.get("path", "")
        if not project_path:
            project_path = scan_project(args.project, args.projects_root).project_path
        path = save_work_graph(project_path, graph)
    if args.persist:
        try:
            repository = SQLiteRepository(args.db or (args.projects_root or "."))
            persist_work_graph(repository, graph.project_id, graph)
        except Exception as exc:
            return _emit_error(args, "persistence_failed", str(exc))
    task_path = None
    if args.generate_tasks:
        scan = scan_project(args.project, args.projects_root)
        start = next_task_number_from_files([
            __import__("os").path.join(scan.project_path, "TASKS.md"),
            __import__("os").path.join(scan.project_path, ".hermes", "tasks.json"),
        ])
        tasks = generate_tasks_from_work_graph(graph, start_at=start)
        task_result = write_task_artifacts(scan.project_path, tasks, overwrite=True)
        task_path = task_result["paths"][0]
    if args.json:
        sys.stdout.write(serialize_work_graph(graph))
    else:
        sys.stdout.write("Work graph: " + graph.project_id + "\n")
        sys.stdout.write("Nodes: " + str(len(graph.nodes)) + "\n")
        sys.stdout.write("Findings: " + str(len(graph.findings)) + "\n")
        if path:
            sys.stdout.write("Wrote: " + path + "\n")
        if task_path:
            sys.stdout.write("Tasks: " + task_path + "\n")
    return WARNING if graph.findings else PASS


def _emit_error(args, code: str, message: str):
    payload = {"error": {"code": code, "message": message}}
    if getattr(args, "json", False):
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        sys.stderr.write("%s: %s\n" % (code, message))
    return INVALID_REQUEST


def _compile(args):
    if not args.template:
        return compile_work_graph(args.project, args.projects_root)
    templates, errors = TemplateLoader().load_path(args.template)
    if errors:
        raise ValueError("; ".join(errors))
    if not templates:
        raise ValueError("No templates loaded from " + args.template)
    graph, error = TemplateCompiler().compile(templates[0], args.project)
    if error:
        raise ValueError(error.message)
    return graph


if __name__ == "__main__":
    raise SystemExit(main())
