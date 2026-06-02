"""CLI entrypoint for tiny package."""
import sys

from tiny_pkg.utils import format_msg


def main():
    """Application entry point."""
    name = sys.argv[1] if len(sys.argv) > 1 else "world"
    print(format_msg(name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
