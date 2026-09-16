import requests
import os
from datetime import datetime

# Notion API Configuration
NOTION_API_URL = "https://api.notion.com/v1/pages"
NOTION_API_KEY = os.getenv("NOTION_API_KEY")  # Set this in your environment variables
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")  # Set this in your environment variables

headers = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}

def classify_content(content):
    """Classify content into categories based on keywords."""
    if "urgent" in content.lower():
        return "Urgent"
    elif "idea" in content.lower():
        return "Idea"
    elif "task" in content.lower():
        return "Task"
    else:
        return "General"

def backup_to_notion(title, content):
    """Backup content to Notion database."""
    category = classify_content(content)
    
    data = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Title": {
                "title": [
                    {"text": {"content": title}}
                ]
            },
            "Category": {
                "select": {"name": category}
            },
            "Date": {
                "date": {"start": datetime.now().isoformat()}
            }
        },
        "children": [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "text": [
                        {"type": "text", "text": {"content": content}}
                    ]
                }
            }
        ]
    }

    response = requests.post(NOTION_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        print("Backup successful!")
    else:
        print(f"Failed to backup: {response.status_code}, {response.text}")

if __name__ == "__main__":
    # Example usage
    title = "Sample Document"
    content = "This is a sample document content. It includes an idea."
    backup_to_notion(title, content)