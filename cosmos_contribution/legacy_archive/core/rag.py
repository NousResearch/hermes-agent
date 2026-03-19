import os
import requests
from bs4 import BeautifulSoup
from utils.config import Config
# from duckduckgo_search import DDGS # Optional for web search

class RAGManager:
    def __init__(self, memory_system):
        self.memory = memory_system
        self.input_dir = os.path.join(os.getcwd(), "data", "inputs")
        os.makedirs(self.input_dir, exist_ok=True)

    def ingest_local_files(self):
        """Ingest text files from data/inputs."""
        count = 0
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".txt") or filename.endswith(".md"):
                path = os.path.join(self.input_dir, filename)
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.memory.add_memory(content, role="system_knowledge", importance=0.8)
                    count += 1
        return f"Ingested {count} files."

    def ingest_url(self, url: str):
        """Scrape and ingest a URL."""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            # Chunking would happen here in a full system
            self.memory.add_memory(text[:5000], role="web_knowledge", importance=0.5) 
            return f"Ingested content from {url}"
        except Exception as e:
            return f"Failed to ingest URL: {e}"

    def recursive_web_search(self, query: str):
        """
        Placeholder for self-refining search strategies.
        In a full build, this would use DDGS to find URLs, then ingest_url them.
        """
        # results = DDGS().text(query, max_results=3)
        # for r in results:
        #     self.ingest_url(r['href'])
        pass
