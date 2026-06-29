import os
import glob

def fix_all_adapters(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith('.py'): continue
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            modified = False
            
            if 'agent.anthropic_adapter' in content:
                content = content.replace('agent.anthropic_adapter', 'agent.providers.anthropic_adapter')
                modified = True
                
            if 'agent.openai_adapter' in content:
                content = content.replace('agent.openai_adapter', 'agent.providers.openai_adapter')
                modified = True
                
            if 'agent.copilot_adapter' in content:
                content = content.replace('agent.copilot_adapter', 'agent.providers.copilot_adapter')
                modified = True
                
            if 'agent.openrouter_adapter' in content:
                content = content.replace('agent.openrouter_adapter', 'agent.providers.openrouter_adapter')
                modified = True
                
            if 'agent.azure_adapter' in content:
                content = content.replace('agent.azure_adapter', 'agent.providers.azure_adapter')
                modified = True
                
            if modified:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Patched {path}")

fix_all_adapters(r'tests\run_agent')
