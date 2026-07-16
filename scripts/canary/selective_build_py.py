"""Keep source-only privileged compatibility shims out of sealed wheels."""

from __future__ import annotations

from setuptools.command.build_py import build_py


_SOURCE_ONLY_MODULES = frozenset({
    ("scripts", "canonical_writer_bootstrap"),
    ("scripts", "canonical_writer_service"),
    ("scripts", "discord_connector_service"),
    ("scripts", "discord_edge_bootstrap"),
    ("scripts", "discord_edge_service"),
    ("scripts.canary", "package_production_runtime_dependencies"),
    ("scripts.canary", "writer_activation"),
})


class SelectiveBuildPy(build_py):
    """Package operational scripts while excluding exact compatibility names."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            item for item in modules if (item[0], item[1]) not in _SOURCE_ONLY_MODULES
        ]
