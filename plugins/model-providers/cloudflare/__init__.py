from typing import Dict, Any, List
from hermes_agent.providers.base import ProviderProfile, register_provider

@register_provider
class CloudflareGatewayProfile(ProviderProfile):
    slug = "cloudflare"
    display_name = "Cloudflare AI Gateway (Unified API)"
    
    required_env_vars = [
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_GATEWAY_NAME",
        "CLOUDFLARE_AIG_TOKEN",       # Für cf-aig-authorization
        "CLOUDFLARE_DYNAMIC_API_KEY"  # Für Authorization: Bearer
    ]
    
    setup_prompts = [
        {
            "key": "CLOUDFLARE_ACCOUNT_ID",
            "prompt": "Gib deine Cloudflare Account ID ein:",
            "type": "string",
            "default": "1d43130db88e4898f15cdb909dc74e8c"
        },
        {
            "key": "CLOUDFLARE_GATEWAY_NAME",
            "prompt": "Gib den Namen deines AI Gateways ein:",
            "type": "string",
            "default": "cfut-gateway"
        },
        {
            "key": "CLOUDFLARE_AIG_TOKEN",
            "prompt": "Gib dein AI Gateway Token ein ($CF_AIG_TOKEN):",
            "type": "secret"
        },
        {
            "key": "CLOUDFLARE_DYNAMIC_API_KEY",
            "prompt": "Gib deinen dynamischen API-Key ein ($DYNAMIC_API_KEY):",
            "type": "secret"
        }
    ]
    
    # Vorschlag für das BitNet-Modell aus deinem curl-Befehl
    fallback_models = [
        "dynamic/Idun-Instruct-VL-BitNet"
    ]

    def get_base_url(self) -> str:
        account = self.config.get("CLOUDFLARE_ACCOUNT_ID", "1d43130db88e4898f15cdb909dc74e8c")
        gateway = self.config.get("CLOUDFLARE_GATEWAY_NAME", "cfut-gateway")
        
        # Wichtig: Der OpenAI-Client von Hermes hängt automatisch /chat/completions an.
        # Deshalb beenden wir die Basis-URL exakt vor dem /compat-Segment.
        return f"https://gateway.ai.cloudflare.com/v1/{account}/{gateway}/compat"

    def get_api_key(self) -> str:
        # Wird in den standardmäßigen 'Authorization: Bearer'-Header injiziert
        return self.config.get("CLOUDFLARE_DYNAMIC_API_KEY")

    def build_api_kwargs_extras(self) -> Dict[str, Any]:
        """
        Injeziert den spezifischen Cloudflare-Sicherheitsheader
        in den Transport-Layer von Hermes.
        """
        aig_token = self.config.get("CLOUDFLARE_AIG_TOKEN")
        
        headers = {}
        if aig_token:
            if not aig_token.startswith("Bearer "):
                aig_token = f"Bearer {aig_token}"
            headers["cf-aig-authorization"] = aig_token
            
        return {"default_headers": headers} 
