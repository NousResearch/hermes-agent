from src.main import main


def test_main_default():
    """Test main returns 0 with no args."""
    assert main() == 0
