from __future__ import annotations

from dataclasses import dataclass, field

from pip._vendor.packaging.utils import NormalizedName, canonicalize_name

from pip._internal.exceptions import CommandError


@dataclass(slots=True)
class ReleaseControl:
    """Helper for managing which release types can be installed."""

    all_releases: set[str] = field(default_factory=set)
    only_final: set[str] = field(default_factory=set)
    _order: list[tuple[str, str]] = field(
        init=False, default_factory=list, compare=False, repr=False
    )

    def handle_mutual_excludes(
        self, value: str, target: set[str], other: set[str], attr_name: str
    ) -> None:
        """Parse and apply release control option value.

        Processes comma-separated package names or special values `:all:` and `:none:`.

        When adding packages to target, they're removed from other to maintain mutual
        exclusivity between all_releases and only_final. All operations are tracked in
        order so that the original command-line argument sequence can be reconstructed
        when passing options to build subprocesses.
        """
        if value.startswith("-"):
            raise CommandError(
                "--all-releases / --only-final option requires 1 argument."
            )
        new = value.split(",")
        while ":all:" in new:
            other.clear()
            target.clear()
            target.add(":all:")
            # Track :all: in order
            self._order.append((attr_name, ":all:"))
            del new[: new.index(":all:") + 1]
            # Without a none, we want to discard everything as :all: covers it
            if ":none:" not in new:
                return
        for name in new:
            if name == ":none:":
                target.clear()
                # Track :none: in order
                self._order.append((attr_name, ":none:"))
                continue
            name = canonicalize_name(name)
            other.discard(name)
            target.add(name)
            # Track package-specific setting in order
            self._order.append((attr_name, name))

    def get_ordered_args(self) -> list[tuple[str, str]]:
        """
        Get ordered list of (flag_name, value) tuples for reconstructing CLI args.

        Returns:
            List of tuples where each tuple is (attribute_name, value).
            The attribute_name is either 'all_releases' or 'only_final'.

        Example:
            [("all_releases", ":all:"), ("only_final", "simple")]
            would be reconstructed as:
            ["--all-releases", ":all:", "--only-final", "simple"]
        """
        return self._order[:]

    def allows_prereleases(self, canonical_name: NormalizedName) -> bool | None:
        """
        Determine if pre-releases are allowed for a package.

        Returns:
            True: Pre-releases are allowed (package in all_releases)
            False: Only final releases allowed (package in only_final)
            None: No specific setting, use default behavior
        """
        if canonical_name in self.all_releases:
            return True
        elif canonical_name in self.only_final:
            return False
        elif ":all:" in self.all_releases:
            return True
        elif ":all:" in self.only_final:
            return False
        return None
