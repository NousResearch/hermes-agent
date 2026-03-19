from core.memory import RecursiveMemory
from core.swarm import SwarmBrain
from core.evolution import EvolutionEngine
from core.rag import RAGManager
from utils.config import Config
import logging

class cosmos:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("cosmos")
        
        self.logger.info("Initializing Hyper-Recursive Memory...")
        self.memory = RecursiveMemory()
        
        self.logger.info("Initializing Agentic Swarm...")
        self.swarm = SwarmBrain()
        
        self.logger.info("Initializing Evolution Engine...")
        self.evolution = EvolutionEngine(self.memory)
        
        self.logger.info("Initializing RAG Manager...")
        self.rag = RAGManager(self.memory)
        
        # Start background processes
        self.evolution.start_background_loop()

    def chat(self, user_input: str) -> str:
        """Main interaction loop."""
        
        # 1. User Co-Evolution: Analyze user style (Simple proxy)
        self.memory.add_memory(user_input, role="user", importance=1.0)
        
        # 2. Retrieve Context
        context = self.memory.retrieve(user_input)
        
        # 3. Agentic Swarm Execution
        # We inject context into the prompt
        enhanced_input = f"Context: {context}\n\nUser: {user_input}"
        response_state = self.swarm.run(enhanced_input)
        
        final_response = response_state['messages'][-1].content
        
        # 4. Save Response
        self.memory.add_memory(final_response, role="cosmos", importance=0.9)
        
        return final_response

    def ingest_data(self, source: str):
        if source.startswith("http"):
            return self.rag.ingest_url(source)
        else:
            return self.rag.ingest_local_files()
