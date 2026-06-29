import os, re

path = 'run_agent.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'    def _is_openrouter_url\(self\) -> bool:[\s\S]*?return hostname == "openrouter\.ai"\n', '', content)
content = re.sub(r'    def _is_direct_openai_url\(self, base_url: str = None\) -> bool:[\s\S]*?return hostname == "api\.openai\.com"\n', '', content)
content = re.sub(r'    def _is_azure_openai_url\(self, base_url: str = None\) -> bool:[\s\S]*?return "openai\.azure\.com" in url\n', '', content)
content = re.sub(r'    def _is_github_copilot_url\(self, base_url: str = None\) -> bool:[\s\S]*?return hostname == "api\.githubcopilot\.com"\n', '', content)
content = re.sub(r'    def _max_tokens_param\(self, value: int\) -> dict:[\s\S]*?return \{"max_tokens": value\}\n', '', content)
content = re.sub(r'    @staticmethod\n    def _requested_output_cap_from_api_kwargs\(api_kwargs: Any\) -> Optional\[int\]:[\s\S]*?return None\n', '', content)
content = re.sub(r'    def _anthropic_prompt_cache_policy\([\s\S]*?return anthropic_prompt_cache_policy\(self, provider=provider, base_url=base_url, api_mode=api_mode, model=model\)\n', '', content)

imports = 'from agent.providers.openai_adapter import is_direct_openai_url, is_azure_openai_url, is_github_copilot_url, max_tokens_param\nfrom agent.providers.openrouter_adapter import is_openrouter_url\nfrom agent.agent_runtime_helpers import anthropic_prompt_cache_policy\n'
if 'is_direct_openai_url' not in content:
    idx = content.find('from agent.iteration_budget import IterationBudget')
    if idx != -1:
        content = content[:idx] + imports + content[idx:]

content = content.replace('self._is_direct_openai_url()', 'is_direct_openai_url(getattr(self, "base_url", None), getattr(self, "_base_url_hostname", ""))')
content = content.replace('self._is_azure_openai_url()', 'is_azure_openai_url(getattr(self, "base_url", None), getattr(self, "_base_url_lower", ""))')
content = content.replace('self._is_github_copilot_url()', 'is_github_copilot_url(getattr(self, "base_url", None), getattr(self, "_base_url_hostname", ""))')
content = content.replace('self._is_openrouter_url()', 'is_openrouter_url(getattr(self, "base_url", None), getattr(self, "_base_url_hostname", ""))')
content = re.sub(r'self\._max_tokens_param\((.*?)\)', r'max_tokens_param(self.model, getattr(self, "base_url", None), getattr(self, "_base_url_hostname", ""), getattr(self, "_base_url_lower", ""), \1)', content)
content = re.sub(r'self\._anthropic_prompt_cache_policy\((.*?)\)', r'anthropic_prompt_cache_policy(self, \1)', content)
content = content.replace('self._anthropic_prompt_cache_policy()', 'anthropic_prompt_cache_policy(self)')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Updated run_agent.py')
