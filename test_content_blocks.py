import sys
import os

# Add the langchain directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'langchain', 'libs', 'core'))

from langchain_core.messages import HumanMessage, AIMessage

# Test case 1: Message with empty string content
def test_empty_string_content():
    msg = HumanMessage(content="")
    blocks = msg.content_blocks
    print("Test 1 - Empty string content:")
    print(f"Content blocks: {blocks}")
    assert blocks == []
    print("✓ Passed\n")

# Test case 2: Message with None content (disabled - Pydantic doesn't allow None content)
# def test_none_content():
#     msg = HumanMessage(content=None)
#     blocks = msg.content_blocks
#     print("Test 2 - None content:")
#     print(f"Content blocks: {blocks}")
#     assert blocks == []
#     print("✓ Passed\n")

# Test case 3: Message with empty list content
def test_empty_list_content():
    msg = HumanMessage(content=[])
    blocks = msg.content_blocks
    print("Test 3 - Empty list content:")
    print(f"Content blocks: {blocks}")
    assert blocks == []
    print("✓ Passed\n")

# Test case 4: Message with normal string content
def test_normal_string_content():
    msg = HumanMessage(content="Hello, world!")
    blocks = msg.content_blocks
    print("Test 4 - Normal string content:")
    print(f"Content blocks: {blocks}")
    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == "Hello, world!"
    print("✓ Passed\n")

# Test case 5: Message with content blocks
def test_content_blocks():
    msg = HumanMessage(content=[
        {"type": "text", "text": "Hello"},
        {"type": "image", "image": {"url": "https://example.com/image.jpg"}}
    ])
    blocks = msg.content_blocks
    print("Test 5 - Content blocks:")
    print(f"Content blocks: {blocks}")
    assert len(blocks) == 2
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == "Hello"
    assert blocks[1]["type"] == "image"
    print("✓ Passed\n")

# Run all tests
if __name__ == "__main__":
    print("Running content blocks tests...\n")
    test_empty_string_content()
    test_none_content()
    test_empty_list_content()
    test_normal_string_content()
    test_content_blocks()
    print("All tests passed! ✓")