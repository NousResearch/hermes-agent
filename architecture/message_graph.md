# Message Graph

```mermaid
graph TD
    %% Message nodes
    SystemMsg["📋 System Message<br/>Role: System<br/>Content: Messages are nodes in a graph"]
    UserMsg["👤 User Message<br/>Role: User<br/>Content: But messages aren't the only thing in the graph"]
    subgraph PrevMessages["Previous Messages"]
        PrevSystemMsg["📋 System Message<br/>Role: System<br/>Content: Edits are kept in the graph as context"]
        PrevUserMsg["👤 User Message<br/>Role: User<br/>Content: So we can ensure they're immutable while keeping them editable"]
    end
    
    %% Chat Response as a subgraph
    subgraph ChatResponseBox["💬 Chat Response"]
        ChatMetadata["📊 Metadata<br/>Temp: 1.0<br/>..."]
        ChatResponseText["📝 Response<br/>Hello, Here's a subagent call: &lt;tool&gt;subagent&lt;/tool&gt;"]
        ChatContent["Content: Hello, Here's a subagent call..."]
    end
    
    %% Tool Response as a subgraph
    subgraph ToolResponseBox["🔧 Tool Response"]
        subgraph ToolMetadata["📊 Tool Metadata"]
            ToolMetadataLength["Length: 3"]
            subgraph ToolChat["💭 Subagent Chat"]
                SubagentSystem["📋 System<br/>Content: Subagent call received"]
                SubagentUser["👤 User<br/>Content: Process this request"]
                SubagentAssistant["🤖 Assistant<br/>Content: Processing..."]
                SubagentSystem --> SubagentUser
                SubagentUser --> SubagentAssistant
            end
        end
        ToolContent["Content: Subagent call output"]
    end
    
    %% Graph flow connections
    SystemMsg --> UserMsg
    PrevSystemMsg --> PrevUserMsg
    PrevMessages -.-> UserMsg
    UserMsg --> ChatResponseBox
    ChatResponseBox --> ToolResponseBox
    
    class SystemMsg,UserMsg messageNode
    class ChatResponseBox responseNode
    class ToolResponseBox responseNode
    class ChatMetadata,ChatResponseText,ChatContent,ToolMetadata,ToolChat,ToolContent,ToolMetadataLength metadataNode
```

Messages should be a graph of immutable elements.

## Why immutable elements?
We want to train on policy
- This means the context cannot change after we call a response.

## Why a graph?
Nodes and connections are a natural way to represent the flow of information in an agent conversation.


## Will this be annoying to deal with?

It shouldn't be! While there will be internal stuff that may look ???, for the interface, it should be as simple as your
normal context window edits, so `message_history[2]['content'] = my_edit`, but internally we'll deal with the recordkeeping
and how this ends up parsing into on policy training data, if requested.