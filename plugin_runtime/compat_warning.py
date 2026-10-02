"""Dependency-free warning contract for plugin compatibility deprecations."""


class HermesPluginCompatWarning(FutureWarning):
    """A plugin imported a name from its pre-decomposition module path."""
