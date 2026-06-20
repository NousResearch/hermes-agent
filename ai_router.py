from enum import Enum
from datetime import datetime
import litellm
class TaskType(Enum):
    FIX = "fix"
    REVIEW = "review"
    TDD = "tdd"
PRICING = {"gpt-4o-mini":{"input":0.15,"output":0.60},"claude-3-5-haiku":{"input":0.80,"output":4.00},"claude-3-5-sonnet":{"input":3.00,"output":15.00}}
ROUTES = {TaskType.FIX:"gpt-4o-mini",TaskType.REVIEW:"claude-3-5-haiku",TaskType.TDD:"claude-3-5-sonnet"}
def call_ai(prompt,task):
    model = ROUTES[task]
    r = litellm.completion(model=model,messages=[{"role":"user","content":prompt}],max_tokens=100)
    return r.choices[0].message.content
if __name__=="__main__":
    print("✅ ai_router.py ready")
