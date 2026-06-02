"""Typer-based CLI application."""
import typer

app = typer.Typer()


@app.command()
def hello(name: str):
    """Say hello."""
    typer.echo(f"Hello {name}")


if __name__ == "__main__":
    app()
