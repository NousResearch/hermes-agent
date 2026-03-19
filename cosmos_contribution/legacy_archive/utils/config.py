import os

class Config:
    # LLM Settings
    OLLAMA_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "llama3.1" # Or 'mistral-nemo'
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Memory Settings
    VECTOR_STORE_PATH = os.path.join(os.getcwd(), "data", "vector_store")
    MEMARORY_DB_PATH = os.path.join(os.getcwd(), "data", "memory.db")
    
    # Evolution Settings
    EVOLUTION_INTERVAL = 3600  # Seconds between evolution cycles
    POPULATION_SIZE = 10
    
    # Swarm Settings
    MAX_SWARM_AGENTS = 5
    
    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.VECTOR_STORE_PATH, exist_ok=True)
        os.makedirs(os.path.dirname(Config.MEMARORY_DB_PATH), exist_ok=True)
