"""Standalone CLI entrypoint."""

from llmwiki_hermes.provider.cli import app


def main() -> None:
    """Run the wiki CLI."""

    app(prog_name="llmwiki-hermes")


if __name__ == "__main__":
    main()
