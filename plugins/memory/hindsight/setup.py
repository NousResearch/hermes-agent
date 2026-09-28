"""Native setup for the employee's separately deployed Hindsight service."""
from pathlib import Path
import sys
from urllib.parse import urlsplit

from hermes_cli.secret_prompt import masked_secret_prompt


def _secret_prompt(label: str) -> str:
    sys.stdout.write(label)
    sys.stdout.flush()
    return masked_secret_prompt("") if sys.stdin.isatty() else sys.stdin.readline().strip()


def save_operational_config(values: dict, hermes_home: str) -> None:
    from gateway.run import _profile_runtime_scope
    from hermes_cli.config import save_config, save_env_value
    settings = {key: str(values[key]).strip() for key in ('url', 'bank_id') if key in values}
    if settings.get('url'):
        url = urlsplit(settings['url'])
        if url.scheme not in ('http', 'https') or not url.netloc:
            raise ValueError('Hindsight URL must be an absolute HTTP or HTTPS URL.')
    with _profile_runtime_scope(Path(hermes_home)):
        save_config({'hindsight': settings, 'memory': {'provider': 'hindsight'}}, merge_existing=True)
        if values.get('api_key'):
            save_env_value('HINDSIGHT_API_KEY', values['api_key'])


def run_setup(provider, hermes_home: str, config: dict) -> None:
    """The native setup owner prepares dependencies before calling this hook."""
    from gateway.run import _profile_runtime_scope
    from hermes_cli.config import load_config_readonly
    with _profile_runtime_scope(Path(hermes_home)):
        current = load_config_readonly().get('hindsight', {})
    print('\n  Configure the deployed Hindsight service. Server inference and memory policy remain fixed.\n')
    default_url = current.get('url') or 'http://127.0.0.1:8888'
    url = input(f'  Hindsight API URL [{default_url}]: ').strip() or default_url
    bank = input(f"  Bank ID [{current.get('bank_id') or 'automatic per profile'}]: ").strip() or current.get('bank_id', '')
    key = _secret_prompt('  Hindsight API key (blank to keep): ')
    provider.save_config({'url': url, 'bank_id': bank, 'api_key': key}, hermes_home)
    print('\n  Hindsight settings saved to native config; keys saved to profile secrets. Start a new session to activate.\n')
