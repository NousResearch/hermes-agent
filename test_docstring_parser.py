import sys
import os

# Add the langchain directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'langchain', 'libs', 'core'))

from langchain_core.utils.function_calling import _parse_google_docstring

# Test case 1: Basic docstring
def test_basic_docstring():
    docstring = """Test function.

    Args:
        arg1: First argument.
        arg2: Second argument.
    """
    args = ["arg1", "arg2"]
    description, arg_descriptions = _parse_google_docstring(docstring, args)
    print("Test 1 - Basic docstring:")
    print(f"Description: {description}")
    print(f"Arg descriptions: {arg_descriptions}")
    assert description == "Test function."
    assert arg_descriptions == {"arg1": "First argument.", "arg2": "Second argument."}
    print("✓ Passed\n")

# Test case 2: Docstring with multi-line descriptions
def test_multi_line_docstring():
    docstring = """Test function with multi-line descriptions.

    This is a longer description that spans multiple lines
    and includes some additional details.

    Args:
        arg1: First argument with a description that
            spans multiple lines and includes more details.
        arg2: Second argument with a short description.
    """
    args = ["arg1", "arg2"]
    description, arg_descriptions = _parse_google_docstring(docstring, args)
    print("Test 2 - Multi-line docstring:")
    print(f"Description: {description}")
    print(f"Arg descriptions: {arg_descriptions}")
    assert description == "Test function with multi-line descriptions. This is a longer description that spans multiple lines and includes some additional details."
    assert arg_descriptions == {
        "arg1": "First argument with a description that spans multiple lines and includes more details.",
        "arg2": "Second argument with a short description."
    }
    print("✓ Passed\n")

# Test case 3: Docstring with type annotations
def test_docstring_with_annotations():
    docstring = """Test function with type annotations.

    Args:
        arg1 (str): First argument as string.
        arg2 (int): Second argument as integer.
    """
    args = ["arg1", "arg2"]
    description, arg_descriptions = _parse_google_docstring(docstring, args)
    print("Test 3 - Docstring with type annotations:")
    print(f"Description: {description}")
    print(f"Arg descriptions: {arg_descriptions}")
    assert description == "Test function with type annotations."
    assert arg_descriptions == {"arg1": "First argument as string.", "arg2": "Second argument as integer."}
    print("✓ Passed\n")

# Test case 4: Docstring with Windows line endings
def test_docstring_with_windows_line_endings():
    docstring = "Test function with Windows line endings.\r\n\r\nArgs:\r\n    arg1: First argument.\r\n    arg2: Second argument.\r\n"
    args = ["arg1", "arg2"]
    description, arg_descriptions = _parse_google_docstring(docstring, args)
    print("Test 4 - Docstring with Windows line endings:")
    print(f"Description: {description}")
    print(f"Arg descriptions: {arg_descriptions}")
    assert description == "Test function with Windows line endings."
    assert arg_descriptions == {"arg1": "First argument.", "arg2": "Second argument."}
    print("✓ Passed\n")

# Test case 5: Docstring with empty lines
def test_docstring_with_empty_lines():
    docstring = """Test function with empty lines.

    Args:

        arg1: First argument.

        arg2: Second argument.

    """
    args = ["arg1", "arg2"]
    description, arg_descriptions = _parse_google_docstring(docstring, args)
    print("Test 5 - Docstring with empty lines:")
    print(f"Description: {description}")
    print(f"Arg descriptions: {arg_descriptions}")
    assert description == "Test function with empty lines."
    assert arg_descriptions == {"arg1": "First argument.", "arg2": "Second argument."}
    print("✓ Passed\n")

# Run all tests
if __name__ == "__main__":
    print("Running docstring parser tests...\n")
    test_basic_docstring()
    test_multi_line_docstring()
    test_docstring_with_annotations()
    test_docstring_with_windows_line_endings()
    test_docstring_with_empty_lines()
    print("All tests passed! ✓")