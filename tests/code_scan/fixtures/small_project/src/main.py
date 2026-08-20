import os
import sys


def main():
    """Application entry point."""
    path = os.getcwd()
    print(f"Working directory: {path}")
    for arg in sys.argv[1:]:
        print(f"  arg: {arg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
