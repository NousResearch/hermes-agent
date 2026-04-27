"""OpenShell sandbox provider plugin."""

from .environment import OpenShellSandboxProvider


def register(ctx):
    ctx.register_sandbox_provider(OpenShellSandboxProvider())
