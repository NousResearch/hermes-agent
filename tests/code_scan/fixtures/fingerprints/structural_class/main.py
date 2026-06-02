import os
import sys
import json


class ConfigParser:
    """Parse configuration files."""

    def __init__(self, path):
        self.path = path
        self.data = {}

    def load(self):
        with open(self.path) as f:
            self.data = json.load(f)

    def get(self, key, default=None):
        return self.data.get(key, default)


class NewClass:
    """Added class — should trigger STRUCTURAL."""
    pass


def main():
    parser = ConfigParser("config.json")
    parser.load()
    value = parser.get("debug", False)
    print(f"Debug mode: {value}")


def helper_utility():
    """A helper function."""
    return os.getcwd()


if __name__ == "__main__":
    main()
