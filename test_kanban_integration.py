import sys
sys.path.insert(0, '/Users/minimetro/.hermes/hermes-agent')
from hermes_agent.tools.kanban_tools import _get_haels_rules_for_task
rules = _get_haels_rules_for_task('operator', None, k=3)
print('Rules for operator:', rules)