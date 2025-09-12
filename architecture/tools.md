# Tools

Not much on this, yet. Tools are just a stateful wrapper around a function, so we can do things like:
- Keep a docker container running
- Keep a game online

```python
class BaseTool:
    def definitions(self) -> List[Dict[str, Any]]:
        # OpenAI API compatible definitions
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        # Returns at minimum {'role': 'tool', 'content': '...'}
        raise NotImplementedError
```