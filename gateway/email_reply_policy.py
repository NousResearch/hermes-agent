"""Optional configured SMTP policy; required-but-unavailable policies refuse delivery."""
from dataclasses import dataclass
from pathlib import Path
import os
import uuid

@dataclass(frozen=True)
class EmailReply:
    sender: str
    recipients: tuple[str, ...]
    subject: str
    body: bytes

class EmailHeld(RuntimeError):
    def __init__(self, reference):
        super().__init__('email held as a local draft, not delivered')
        self.reference=reference

def enforce_email_reply(message):
    from hermes_cli.config import load_config
    from hermes_cli.plugins import get_plugin_manager
    from hermes_constants import get_hermes_home
    name=load_config().get('email', {}).get('required_reply_policy')
    if name is None: return
    if not isinstance(name,str) or not name: raise RuntimeError('invalid required email policy')
    manager=get_plugin_manager();manager.discover_and_load()
    provider=manager._email_reply_policies.get(name)
    if provider is None: raise RuntimeError('required email policy unavailable')
    from email.utils import getaddresses
    addresses=tuple(addr for _,addr in getaddresses(message.get_all('To', [])+message.get_all('Cc', [])+message.get_all('Bcc', [])))
    envelope=EmailReply(str(message.get('From','')),addresses,str(message.get('Subject','')),message.as_bytes())
    decision=provider.evaluate(envelope)
    if manager._email_reply_policies.get(name) is not provider: raise RuntimeError('email policy changed')
    if decision=='allow': return
    if decision!='hold': raise RuntimeError('email policy denied or invalid')
    drafts=Path(get_hermes_home())/'drafts';drafts.mkdir(mode=0o700,parents=True,exist_ok=True)
    reference=uuid.uuid4().hex
    fd=os.open(drafts/(reference+'.eml'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(envelope.body);stream.flush();os.fsync(stream.fileno())
    raise EmailHeld(reference)
