"""CLI entrypoint for compiling Hermes OS work graphs."""

import argparse
import json
import sys

from .work_graph import compile_work_graph, save_work_graph, serialize_work_graph


PASS = 0
WARNING = 2
INVALID_REQUEST = 64


def build_parser():
    parser = argparse.ArgumentParser(prog="hermes plan")
    parser.add_argument("project")
    parser.add_argument("--projects-root")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.project:
        return INVALID_REQUEST
    graph = compile_work_graph(args.project, args.projects_root)
    path = None
    if args.write and graph.nodes:
        project_path = graph.nodes[0].metadata.get("path", "")
        path = save_work_graph(project_path, graph)
    if args.json:
        sys.stdout.write(serialize_work_graph(graph))
    else:
        sys.stdout.write("Work graph: " + graph.project_id + "\n")
        sys.stdout.write("Nodes: " + str(len(graph.nodes)) + "\n")
        sys.stdout.write("Findings: " + str(len(graph.findings)) + "\n")
        if path:
            sys.stdout.write("Wrote: " + path + "\n")
    return WARNING if graph.findings else PASS


if __name__ == "__main__":
    raise SystemExit(main())
