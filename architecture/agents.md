# Agents

Agents can be viewed as an FSM using an LLM to generate inputs into the system that operates over a DAG.

What this really means is that the agent is just a function without memory that uses text inputs and outputs in a
defined order.

```python
def my_agent(*args, **kwargs) -> str:
    # do whatever you want!
    return "Hi I'm an agent!"
```

Now obviously, that's like saying water's wet, but we're going to be using that definition to inform our design of the
library, namely, that we should *not* store agent state outside the function call.

## The Agent Class

So we don't have state, why are we using a class?

Well, we want to initialize things, we want to have some configuration, and we want to have some helper functions.
Preferably all in a single place.

```python
class BaseAgent:
    def agent_primitives(self) -> list[BaseAgent]:
        # Returns a list of Agents that are utilized by this agent to generate inputs
        # We use agent primitives here instead of subagents because these are going to be part
        # of the message graph, not a subagent tool call.
        raise NotImplementedError
    
    def tools(self) -> list[BaseTool]:
        # Returns a list of tools that the agent needs to run
        raise NotImplementedError
    
    
    def run(self, config, *args, **kwargs) -> ConversationGraph:
        llm = get_llm(config)
        tools = self.tools()
        for agent in self.agent_primitives():
            tools.extend(agent.tools())
        tools = remove_duplicates(tools)
        tools = initialize_tools(tools, config)
        return self(llm, tools, config, *args, **kwargs)
    
    @staticmethod
    def __call__(self, llm, tools, config, *args, **kwargs) -> ConversationGraph:
        # Returns a ConversationGraph that can be parsed to get the output of the agent
        # Use w/e args/kwargs you want, as long as llm/tools/config are satisfied. 
        raise NotImplementedError
```

Doesn't seem too bad (I hope), it is a bit annoying that we don't initialize everything in the constructor, but
hopefully we all kinda like it :)

