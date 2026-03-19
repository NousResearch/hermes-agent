from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from utils.config import Config
import operator

# Define Swarm State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_step: str
    subtasks: List[str]
    current_task_idx: int
    criticism: str

class SwarmBrain:
    def __init__(self):
        self.llm = ChatOllama(model=Config.DEFAULT_MODEL, base_url=Config.OLLAMA_BASE_URL, temperature=0.7)
        self.workflow = StateGraph(AgentState)
        self._build_graph()
    
    def _build_graph(self):
        self.workflow.add_node("planner", self.plan_node)
        self.workflow.add_node("executor", self.execute_node)
        self.workflow.add_node("critic", self.critic_node)
        
        self.workflow.set_entry_point("planner")
        
        self.workflow.add_conditional_edges(
            "planner",
            lambda x: "executor" if x['subtasks'] else END
        )
        self.workflow.add_edge("executor", "critic")
        self.workflow.add_conditional_edges(
            "critic",
            lambda x: "executor" if x['next_step'] == "retry" else "planner"
        )
        
        self.app = self.workflow.compile()

    def plan_node(self, state: AgentState):
        """Marge-style reflective decomposition."""
        last_msg = state['messages'][-1].content
        # Simple prompt to decompose
        if not state.get('subtasks'):
            prompt = f"Break down this task into 3 distinct steps: {last_msg}. Return comma separated list."
            response = self.llm.invoke(prompt)
            tasks = [t.strip() for t in response.content.split(',')]
            return {"subtasks": tasks, "current_task_idx": 0, "next_step": "execute"}
        
        # If returning from execution, check if done
        if state['current_task_idx'] >= len(state['subtasks']):
             return {"next_step": END}
        
        return {"next_step": "execute"}

    def execute_node(self, state: AgentState):
        """Ralph-style execution."""
        idx = state['current_task_idx']
        if idx >= len(state['subtasks']):
            return {"next_step": "critic"}
            
        task = state['subtasks'][idx]
        # Simulate execution or use tools here
        result = self.llm.invoke(f"Execute this subtask briefly: {task}")
        return {
            "messages": [AIMessage(content=f"Task {idx+1} Result: {result.content}")],
            "current_task_idx": idx + 1
        }

    def critic_node(self, state: AgentState):
        """Self-verification."""
        last_result = state['messages'][-1].content
        audit = self.llm.invoke(f"Critique this result: {last_result}. Is it satisfactory? Yes/No.")
        if "no" in audit.content.lower():
            # In a real loom, we would decrement idx and retry
            return {"criticism": audit.content, "next_step": "retry"} 
        return {"criticism": "Approved", "next_step": "continue"}

    def run(self, user_input: str):
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "subtasks": [],
            "current_task_idx": 0,
            "next_step": "plan",
            "criticism": ""
        }
        return self.app.invoke(initial_state)
