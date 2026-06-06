# Next Env Example Remediation Queue

Review these keys before editing real `.env.example` files.

| Project | Key | Status | Action | Secret | Confidence |
|---|---|---|---|---|---|
| 500K Project | `GRAFANA_BASE_URL` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `GRAFANA_COMPARE_ALL_DASHBOARD` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `GRAFANA_DISK_PER_PROJECT_DASHBOARD` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `LINUXNAT_HOST` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `LINUXNAT_PORT` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `LINUXNAT_SSH_CONFIG_NAME` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `LINUXNAT_SUDO_PASSWORD` | unused_env | review_remove_or_document | yes | medium |
| 500K Project | `LINUXNAT_USER` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `LXC_MONITORING_ID` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `LXC_WAZUH_ID` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `PROXMOX_KERNEL` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `PROXMOX_MAX_CPU` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `PROXMOX_MAX_MEM_GIB` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `PROXMOX_NODE_NAME` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `PROXMOX_STORAGE_BACKUP` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `PROXMOX_STORAGE_LOCAL_ZFS` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `PROXMOX_VERSION` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `SSH_CONFIG_NAME` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `SSH_HOST` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `SSH_KEY_PATH` | unused_env | review_remove_or_document | yes | medium |
| 500K Project | `SSH_PORT` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `SSH_USER` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `VM_GITLAB_ID` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `VM_LINUX_DEV_ID` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `VM_LINUX_NAT_ID` | unused_env | review_remove_or_document | no | medium |
| 500K Project | `VM_WINSERVER_ID` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `AI_AUTO_ROUTING_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `AI_DEFAULT_PROVIDER` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `ANTHROPIC_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Jigsaw AI Team Mock 2026 | `ANTHROPIC_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `ANTHROPIC_MARKUP` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `APP_ENV` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `APP_HOST` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `APP_PORT` | review | review_context | no | low |
| Jigsaw AI Team Mock 2026 | `BACKUP_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `BACKUP_SCHEDULE` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `CORS_ALLOWED_ORIGINS` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `CSRF_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `CSRF_SECRET` | unused_env | review_remove_or_document | yes | medium |
| Jigsaw AI Team Mock 2026 | `DATABASE_URL` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `ENVIRONMENT` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `EXECUTIONS_DATA_HARD_DELETE_BUFFER` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `EXECUTIONS_DATA_PRUNE` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `EXECUTIONS_DATA_SAVE_MANUAL_EXECUTIONS` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `EXECUTIONS_DATA_SAVE_ON_ERROR` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `EXECUTIONS_DATA_SAVE_ON_SUCCESS` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `EXECUTIONS_MODE` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `FRONTEND_URL` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `GEMINI_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Jigsaw AI Team Mock 2026 | `GEMINI_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `GEMINI_MARKUP` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `JWT_ACCESS_TOKEN_EXPIRY` | unused_env | review_remove_or_document | yes | medium |
| Jigsaw AI Team Mock 2026 | `JWT_EXPIRY` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `JWT_REFRESH_TOKEN_EXPIRY` | unused_env | review_remove_or_document | yes | medium |
| Jigsaw AI Team Mock 2026 | `JWT_SECRET` | review | review_context | yes | low |
| Jigsaw AI Team Mock 2026 | `JWT_USE_RSA` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `MINIO_PORT` | review | review_context | no | low |
| Jigsaw AI Team Mock 2026 | `MINIO_ROOT_PASSWORD` | review | review_context | yes | low |
| Jigsaw AI Team Mock 2026 | `MINIO_ROOT_USER` | review | review_context | no | low |
| Jigsaw AI Team Mock 2026 | `N8N_AVAILABLE_BINARY_DATA_MODES` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_BASIC_AUTH_ACTIVE` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_DB_POSTGRESDB_HOST` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_DB_POSTGRESDB_PORT` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_DB_TYPE` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_DEFAULT_BINARY_DATA_MODE` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_DIAGNOSTICS_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_EXECUTIONS_RETRY_ATTEMPTS` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_EXECUTIONS_RETRY_WAIT` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_HIRING_BANNER_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_HOST` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_LOG_FILE_COUNT_MAX` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_LOG_FILE_LOCATION` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_LOG_FILE_SIZE_MAX` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_PATH` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_PERSONALIZATION_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_PROTOCOL` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_PUBLIC_API_DISABLED` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_SECURE_COOKIE` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `N8N_VERSION_NOTIFICATIONS_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `NODE_OPTIONS` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `OPENAI_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Jigsaw AI Team Mock 2026 | `OPENAI_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `OPENAI_MARKUP` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `OPENROUTER_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `OPENROUTER_MARKUP` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `POSTGRES_SSLMODE` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `QUEUE_BULL_REDIS_DB` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `QUEUE_BULL_REDIS_HOST` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `QUEUE_BULL_REDIS_PORT` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `QUEUE_HEALTH_CHECK_ACTIVE` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `RATE_LIMIT_API` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `RATE_LIMIT_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `RATE_LIMIT_LOGIN` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `REDIS_DB` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `REDIS_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `REDIS_HOST` | review | review_context | no | low |
| Jigsaw AI Team Mock 2026 | `REDIS_URL` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `SERVER_HOST` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `SERVER_PORT` | unused_env | review_remove_or_document | no | medium |
| Jigsaw AI Team Mock 2026 | `TZ` | unused_env | review_remove_or_document | no | medium |
| MD Assist by AI | `ASANA_CLIENT_ID` | review | review_context | no | low |
| MD Assist by AI | `ASANA_CLIENT_SECRET` | review | review_context | yes | low |
| MD Assist by AI | `ASANA_DEFAULT_PROJECT_ID` | review | review_context | no | low |
| MD Assist by AI | `DEEPSEEK_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| MD Assist by AI | `GITHUB_DEFAULT_ORG` | review | review_context | no | low |
| MD Assist by AI | `GITHUB_PAT` | review | review_context | no | low |
| MD Assist by AI | `GITLAB_TOKEN` | unused_env | review_remove_or_document | yes | medium |
| MD Assist by AI | `GITLAB_URL` | unused_env | review_remove_or_document | no | medium |
| MD Assist by AI | `GOOGLE_CLIENT_ID` | review | review_context | no | low |
| MD Assist by AI | `GOOGLE_CLIENT_SECRET` | review | review_context | yes | low |
| MD Assist by AI | `GOOGLE_DRIVE_ROOT_FOLDER_ID` | review | review_context | no | low |
| MD Assist by AI | `GOOGLE_REFRESH_TOKEN` | review | review_context | yes | low |
| MD Assist by AI | `LARK_BASE_APP_TOKEN` | unused_env | review_remove_or_document | yes | medium |
| MD Assist by AI | `LARK_CHAT_MD_DM` | review | review_context | no | low |
| MD Assist by AI | `LARK_ROUTINE_BASE_APP_TOKEN` | review | review_context | yes | low |
| MD Assist by AI | `LARK_TABLE_CASHFLOW` | unused_env | review_remove_or_document | no | medium |
| MD Assist by AI | `LARK_TENANT_KEY` | review | review_context | yes | low |
| MD Assist by AI | `LINUXNAT_HOST` | unused_env | review_remove_or_document | no | medium |
| MD Assist by AI | `LINUXNAT_PORT` | unused_env | review_remove_or_document | no | medium |
| MD Assist by AI | `LINUXNAT_SSH_CONFIG_NAME` | unused_env | review_remove_or_document | no | medium |
| MD Assist by AI | `LINUXNAT_SUDO_PASSWORD` | unused_env | review_remove_or_document | yes | medium |
| MD Assist by AI | `LINUXNAT_USER` | unused_env | review_remove_or_document | no | medium |
| MD Assist by AI | `MD_EMAIL` | review | review_context | no | low |
| MD Assist by AI | `MD_NAME` | review | review_context | no | low |
| MD Assist by AI | `MD_PHONE` | review | review_context | no | low |
| MD Assist by AI | `MOONSHOT_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| MD Assist by AI | `OPENAI_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| MD Assist by AI | `VPS_DEPLOY_PATH` | review | review_context | no | low |
| MD Assist by AI | `VPS_SSH_KEY` | unused_env | review_remove_or_document | yes | medium |
| MD Assist by AI | `VPS_SSH_PORT` | unused_env | review_remove_or_document | no | medium |
| Master SynerryEoffice | `AI_GATEWAY_DEFAULT_MODEL` | review | review_context | no | low |
| Master SynerryEoffice | `DATABASE_URL` | review | review_context | no | low |
| Master SynerryEoffice | `GITLAB_TOKEN` | review | review_context | yes | low |
| Master SynerryEoffice | `JWT_EXPIRY` | review | review_context | no | low |
| Master SynerryEoffice | `JWT_SECRET` | review | review_context | yes | low |
| Master SynerryEoffice | `JWT_SENSITIVE_EXPIRY` | review | review_context | no | low |
| Master SynerryEoffice | `LINUXNAT_HOST` | unused_env | review_remove_or_document | no | medium |
| Master SynerryEoffice | `LINUXNAT_PORT` | unused_env | review_remove_or_document | no | medium |
| Master SynerryEoffice | `LINUXNAT_SSH_CONFIG_NAME` | unused_env | review_remove_or_document | no | medium |
| Master SynerryEoffice | `LINUXNAT_SUDO_PASSWORD` | unused_env | review_remove_or_document | yes | medium |
| Master SynerryEoffice | `LINUXNAT_USER` | unused_env | review_remove_or_document | no | medium |
| Master SynerryEoffice | `MFA_ENCRYPTION_KEY` | review | review_context | yes | low |
| Master SynerryEoffice | `MFA_ISSUER` | review | review_context | no | low |
| Master SynerryEoffice | `OPENCLAW_DEFAULT_MODEL` | unused_env | review_remove_or_document | no | medium |
| Master SynerryEoffice | `OPENCLAW_GATEWAY_PORT` | unused_env | review_remove_or_document | no | medium |
| Master SynerryEoffice | `OPENROUTER_API_KEY` | review | review_context | yes | low |
| Master SynerryEoffice | `OPENROUTER_MODEL` | unused_env | review_remove_or_document | no | medium |
| Master SynerryEoffice | `REDIS_URL` | review | review_context | no | low |
| Master SynerryEoffice | `VPS_SSH_KEY` | review | review_context | yes | low |
| Master SynerryEoffice | `VPS_USER` | review | review_context | no | low |
| Master SynerryNew | `DEV_URL` | review | review_context | no | low |
| Master SynerryNew | `GITLAB_TOKEN` | review | review_context | yes | low |
| Master SynerryNew | `NEXT_PUBLIC_SITE_NAME` | review | review_context | no | low |
| Master SynerryNew | `NEXT_PUBLIC_SITE_URL` | review | review_context | no | low |
| Master SynerryNew | `OPENCLAW_DEFAULT_MODEL` | unused_env | review_remove_or_document | no | medium |
| Master SynerryNew | `OPENCLAW_GATEWAY_PORT` | unused_env | review_remove_or_document | no | medium |
| Master SynerryNew | `OPENROUTER_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Master SynerryNew | `SMTP_FROM` | review | review_context | no | low |
| Master SynerryNew | `SMTP_HOST` | review | review_context | no | low |
| Master SynerryNew | `SMTP_PASS` | review | review_context | no | low |
| Master SynerryNew | `SMTP_PORT` | review | review_context | no | low |
| Master SynerryNew | `SMTP_TO` | review | review_context | no | low |
| Master SynerryNew | `SMTP_USER` | review | review_context | no | low |
| Master SynerryNew | `VPS_PASSWORD` | unused_env | review_remove_or_document | yes | medium |
| Master SynerryNew | `VPS_SSH_KEY` | unused_env | review_remove_or_document | yes | medium |
| Master SynerryNew | `VPS_USER` | review | review_context | no | low |
| Master WebEngine | `AI_DEFAULT_MODEL` | review | review_context | no | low |
| Master WebEngine | `AI_FALLBACK_MODEL` | review | review_context | no | low |
| Master WebEngine | `AI_GATEWAY_DEFAULT_MODEL` | review | review_context | no | low |
| Master WebEngine | `AI_MAX_TOKENS` | review | review_context | yes | low |
| Master WebEngine | `AI_RATE_LIMIT_PER_MIN` | review | review_context | no | low |
| Master WebEngine | `AI_TOKEN_MARKUP` | review | review_context | yes | low |
| Master WebEngine | `API_HOST` | review | review_context | no | low |
| Master WebEngine | `BOI_FEED_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Master WebEngine | `BOI_MAP_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Master WebEngine | `BOI_MATCHING_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Master WebEngine | `BOI_SOURCING_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Master WebEngine | `BOI_SUPPLIERS_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Master WebEngine | `DEFAULT_LANG` | unused_env | review_remove_or_document | no | medium |
| Master WebEngine | `DRA_HAJJ_ENABLED` | unused_env | review_remove_or_document | no | medium |
| Master WebEngine | `DRA_RELIGION_SUBSITES` | unused_env | review_remove_or_document | no | medium |
| Master WebEngine | `ES_JAVA_OPTS` | unused_env | review_remove_or_document | no | medium |
| Master WebEngine | `GITLAB_DEPLOY_TOKEN` | review | review_context | yes | low |
| Master WebEngine | `GITLAB_DEPLOY_USER` | review | review_context | no | low |
| Master WebEngine | `JWT_EXPIRES_IN` | review | review_context | no | low |
| Master WebEngine | `JWT_REFRESH_EXPIRES_IN` | review | review_context | no | low |
| Master WebEngine | `MAX_FILE_SIZE` | review | review_context | no | low |
| Master WebEngine | `MINIMAX_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Master WebEngine | `MINIO_ACCESS_KEY` | review | review_context | yes | low |
| Master WebEngine | `MINIO_BUCKET` | review | review_context | no | low |
| Master WebEngine | `MINIO_ENDPOINT` | review | review_context | no | low |
| Master WebEngine | `MINIO_PORT` | review | review_context | no | low |
| Master WebEngine | `MINIO_SECRET_KEY` | review | review_context | yes | low |
| Master WebEngine | `MMX_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Master WebEngine | `NEXTAUTH_URL` | unused_env | review_remove_or_document | no | medium |
| Master WebEngine | `POSTGRES_DB` | unused_env | review_remove_or_document | no | medium |
| Master WebEngine | `POSTGRES_USER` | unused_env | review_remove_or_document | no | medium |
| Master WebEngine | `REDIS_PASSWORD` | unused_env | review_remove_or_document | yes | medium |
| Master WebEngine | `SEARCH_ENGINE` | review | review_context | no | low |
| Master WebEngine | `STAGING_SSH_HOST` | review | review_context | no | low |
| Master WebEngine | `STAGING_SSH_PASS` | review | review_context | no | low |
| Master WebEngine | `STAGING_SSH_USER` | review | review_context | no | low |
| Master WebEngine | `SUPPORTED_LANGUAGES` | unused_env | review_remove_or_document | no | medium |
| Master WebEngine | `VPS_SSH_KEY` | review | review_context | yes | low |
| Master WebEngine | `VPS_USER` | review | review_context | no | low |
| Master WebEngine-mab-main | `AI_DEFAULT_MODEL` | review | review_context | no | low |
| Master WebEngine-mab-main | `AI_FALLBACK_MODEL` | review | review_context | no | low |
| Master WebEngine-mab-main | `AI_GATEWAY_DEFAULT_MODEL` | review | review_context | no | low |
| Master WebEngine-mab-main | `AI_MAX_TOKENS` | review | review_context | yes | low |
| Master WebEngine-mab-main | `AI_RATE_LIMIT_PER_MIN` | review | review_context | no | low |
| Master WebEngine-mab-main | `AI_TOKEN_MARKUP` | review | review_context | yes | low |
| Master WebEngine-mab-main | `API_HOST` | review | review_context | no | low |
| Master WebEngine-mab-main | `GITLAB_DEPLOY_TOKEN` | review | review_context | yes | low |
| Master WebEngine-mab-main | `GITLAB_DEPLOY_USER` | review | review_context | no | low |
| Master WebEngine-mab-main | `JWT_EXPIRES_IN` | review | review_context | no | low |
| Master WebEngine-mab-main | `JWT_REFRESH_EXPIRES_IN` | review | review_context | no | low |
| Master WebEngine-mab-main | `MAX_FILE_SIZE` | review | review_context | no | low |
| Master WebEngine-mab-main | `MINIMAX_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Master WebEngine-mab-main | `MINIO_ACCESS_KEY` | review | review_context | yes | low |
| Master WebEngine-mab-main | `MINIO_BUCKET` | review | review_context | no | low |
| Master WebEngine-mab-main | `MINIO_ENDPOINT` | review | review_context | no | low |
| Master WebEngine-mab-main | `MINIO_PORT` | review | review_context | no | low |
| Master WebEngine-mab-main | `MINIO_SECRET_KEY` | review | review_context | yes | low |
| Master WebEngine-mab-main | `MMX_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Master WebEngine-mab-main | `SEARCH_ENGINE` | review | review_context | no | low |
| Master WebEngine-mab-main | `STAGING_SSH_HOST` | review | review_context | no | low |
| Master WebEngine-mab-main | `STAGING_SSH_PASS` | review | review_context | no | low |
| Master WebEngine-mab-main | `STAGING_SSH_USER` | review | review_context | no | low |
| Master WebEngine-mab-main | `VPS_SSH_KEY` | review | review_context | yes | low |
| Master WebEngine-mab-main | `VPS_USER` | review | review_context | no | low |
| Master WebEngine-mab-vps | `AI_DEFAULT_MODEL` | review | review_context | no | low |
| Master WebEngine-mab-vps | `AI_FALLBACK_MODEL` | review | review_context | no | low |
| Master WebEngine-mab-vps | `AI_GATEWAY_DEFAULT_MODEL` | review | review_context | no | low |
| Master WebEngine-mab-vps | `AI_MAX_TOKENS` | review | review_context | yes | low |
| Master WebEngine-mab-vps | `AI_RATE_LIMIT_PER_MIN` | review | review_context | no | low |
| Master WebEngine-mab-vps | `AI_TOKEN_MARKUP` | review | review_context | yes | low |
| Master WebEngine-mab-vps | `API_HOST` | review | review_context | no | low |
| Master WebEngine-mab-vps | `GITLAB_DEPLOY_TOKEN` | review | review_context | yes | low |
| Master WebEngine-mab-vps | `GITLAB_DEPLOY_USER` | review | review_context | no | low |
| Master WebEngine-mab-vps | `JWT_EXPIRES_IN` | review | review_context | no | low |
| Master WebEngine-mab-vps | `JWT_REFRESH_EXPIRES_IN` | review | review_context | no | low |
| Master WebEngine-mab-vps | `MAX_FILE_SIZE` | review | review_context | no | low |
| Master WebEngine-mab-vps | `MINIMAX_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Master WebEngine-mab-vps | `MINIO_ACCESS_KEY` | review | review_context | yes | low |
| Master WebEngine-mab-vps | `MINIO_BUCKET` | review | review_context | no | low |
| Master WebEngine-mab-vps | `MINIO_ENDPOINT` | review | review_context | no | low |
| Master WebEngine-mab-vps | `MINIO_PORT` | review | review_context | no | low |
| Master WebEngine-mab-vps | `MINIO_SECRET_KEY` | review | review_context | yes | low |
| Master WebEngine-mab-vps | `MMX_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Master WebEngine-mab-vps | `SEARCH_ENGINE` | review | review_context | no | low |
| Master WebEngine-mab-vps | `STAGING_SSH_HOST` | review | review_context | no | low |
| Master WebEngine-mab-vps | `STAGING_SSH_PASS` | review | review_context | no | low |
| Master WebEngine-mab-vps | `STAGING_SSH_USER` | review | review_context | no | low |
| Master WebEngine-mab-vps | `VPS_SSH_KEY` | review | review_context | yes | low |
| Master WebEngine-mab-vps | `VPS_USER` | review | review_context | no | low |
| Support Center | `BW_CLIENTID` | review | review_context | no | low |
| Support Center | `BW_CLIENTSECRET` | review | review_context | yes | low |
| Support Center | `DATABASE_URL` | review | review_context | no | low |
| Support Center | `LOG_LEVEL` | review | review_context | no | low |
| Support Center | `UI_PORT` | review | review_context | no | low |
| dra-merge-worktree | `AI_DEFAULT_MODEL` | review | review_context | no | low |
| dra-merge-worktree | `AI_FALLBACK_MODEL` | review | review_context | no | low |
| dra-merge-worktree | `AI_GATEWAY_DEFAULT_MODEL` | review | review_context | no | low |
| dra-merge-worktree | `AI_MAX_TOKENS` | review | review_context | yes | low |
| dra-merge-worktree | `AI_RATE_LIMIT_PER_MIN` | review | review_context | no | low |
| dra-merge-worktree | `AI_TOKEN_MARKUP` | review | review_context | yes | low |
| dra-merge-worktree | `API_HOST` | review | review_context | no | low |
| dra-merge-worktree | `GITLAB_DEPLOY_TOKEN` | review | review_context | yes | low |
| dra-merge-worktree | `GITLAB_DEPLOY_USER` | review | review_context | no | low |
| dra-merge-worktree | `JWT_EXPIRES_IN` | review | review_context | no | low |
| dra-merge-worktree | `JWT_REFRESH_EXPIRES_IN` | review | review_context | no | low |
| dra-merge-worktree | `MAX_FILE_SIZE` | review | review_context | no | low |
| dra-merge-worktree | `MINIO_ACCESS_KEY` | review | review_context | yes | low |
| dra-merge-worktree | `MINIO_BUCKET` | review | review_context | no | low |
| dra-merge-worktree | `MINIO_ENDPOINT` | review | review_context | no | low |
| dra-merge-worktree | `MINIO_PORT` | review | review_context | no | low |
| dra-merge-worktree | `MINIO_SECRET_KEY` | review | review_context | yes | low |
| dra-merge-worktree | `SEARCH_ENGINE` | review | review_context | no | low |
| dra-merge-worktree | `STAGING_SSH_HOST` | review | review_context | no | low |
| dra-merge-worktree | `STAGING_SSH_PASS` | review | review_context | no | low |
| dra-merge-worktree | `STAGING_SSH_USER` | review | review_context | no | low |
| dra-merge-worktree | `VPS_SSH_KEY` | review | review_context | yes | low |
| dra-merge-worktree | `VPS_USER` | review | review_context | no | low |
| EA Factoring | `AI_MONTHLY_BUDGET_USD` | review | review_context | no | low |
| EA Factoring | `NODE_ENV` | review | review_context | no | low |
| EA Factoring | `REDIS_URL` | review | review_context | no | low |
| Idea2Logic | `FREEPIK_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Idea2Logic | `LINUXNAT_HOST` | unused_env | review_remove_or_document | no | medium |
| Idea2Logic | `LINUXNAT_PORT` | unused_env | review_remove_or_document | no | medium |
| Idea2Logic | `LINUXNAT_SSH_CONFIG_NAME` | unused_env | review_remove_or_document | no | medium |
| Idea2Logic | `LINUXNAT_SUDO_PASSWORD` | unused_env | review_remove_or_document | yes | medium |
| Idea2Logic | `LINUXNAT_USER` | unused_env | review_remove_or_document | no | medium |
| Idea2Logic | `SUPABASE_AUTH_EXTERNAL_GOOGLE_CLIENT_ID` | unused_env | review_remove_or_document | no | medium |
| Idea2Logic | `SUPABASE_AUTH_EXTERNAL_GOOGLE_SECRET` | unused_env | review_remove_or_document | yes | medium |
| MQ5 Market | `AI_MONTHLY_BUDGET_USD` | review | review_context | no | low |
| MQ5 Market | `CAPITAL_IDENTIFIER` | unused_env | review_remove_or_document | no | medium |
| MQ5 Market | `CLOUDFLARE_API_TOKEN` | unused_env | review_remove_or_document | yes | medium |
| MQ5 Market | `DATABASE_URL` | review | review_context | no | low |
| MQ5 Market | `VPS_SSH_PORT` | unused_env | review_remove_or_document | no | medium |
| MQ5 Market | `WEB_PORT` | review | review_context | no | low |
| Master AdsPilot-AI | `ANTHROPIC_API_KEY` | review | review_context | yes | low |
| Master AdsPilot-AI | `API_URL` | review | review_context | no | low |
| Master AdsPilot-AI | `APP_URL` | review | review_context | no | low |
| Master AdsPilot-AI | `DATABASE_URL` | review | review_context | no | low |
| Master AdsPilot-AI | `ENCRYPTION_KEY` | review | review_context | yes | low |
| Master AdsPilot-AI | `GITLAB_TOKEN` | review | review_context | yes | low |
| Master AdsPilot-AI | `GOOGLE_CLIENT_ID` | review | review_context | no | low |
| Master AdsPilot-AI | `GOOGLE_CLIENT_SECRET` | review | review_context | yes | low |
| Master AdsPilot-AI | `JWT_EXPIRY` | review | review_context | no | low |
| Master AdsPilot-AI | `JWT_SECRET` | review | review_context | yes | low |
| Master AdsPilot-AI | `LINE_NOTIFY_TOKEN` | review | review_context | yes | low |
| Master AdsPilot-AI | `META_APP_ID` | review | review_context | no | low |
| Master AdsPilot-AI | `META_APP_SECRET` | review | review_context | yes | low |
| Master AdsPilot-AI | `OPENAI_API_KEY` | review | review_context | yes | low |
| Master AdsPilot-AI | `OPENROUTER_API_KEY` | review | review_context | yes | low |
| Master AdsPilot-AI | `REDIS_URL` | review | review_context | no | low |
| Master AdsPilot-AI | `SLACK_WEBHOOK_URL` | review | review_context | no | low |
| Master AdsPilot-AI | `TIMESCALE_URL` | review | review_context | no | low |
| Master AdsPilot-AI | `VPS_SSH_KEY` | review | review_context | yes | low |
| Master AdsPilot-AI | `VPS_USER` | review | review_context | no | low |
| Master Content Factory | `AUTH_GOOGLE_ID` | review | review_context | no | low |
| Master Content Factory | `AUTH_GOOGLE_SECRET` | review | review_context | yes | low |
| Master Content Factory | `AUTH_LINE_CHANNEL_ID` | review | review_context | no | low |
| Master Content Factory | `AUTH_LINE_CHANNEL_SECRET` | review | review_context | yes | low |
| Master Content Factory | `AUTH_SECRET` | review | review_context | yes | low |
| Master Content Factory | `AUTH_URL` | review | review_context | no | low |
| Master Content Factory | `BINANCE_API_KEY` | review | review_context | yes | low |
| Master Content Factory | `BINANCE_SECRET_KEY` | review | review_context | yes | low |
| Master Content Factory | `CF_FB_PAGE_ACCESS_TOKEN` | review | review_context | yes | low |
| Master Content Factory | `CF_FB_PAGE_ID` | review | review_context | no | low |
| Master Content Factory | `CF_HOST` | review | review_context | no | low |
| Master Content Factory | `CF_LINKEDIN_ACCESS_TOKEN` | review | review_context | yes | low |
| Master Content Factory | `CF_MINIMAX_VOICE_ID_EN` | review | review_context | no | low |
| Master Content Factory | `CF_MINIMAX_VOICE_ID_TH` | review | review_context | no | low |
| Master Content Factory | `CF_MINIO_ACCESS_KEY` | review | review_context | yes | low |
| Master Content Factory | `CF_MINIO_BUCKET` | review | review_context | no | low |
| Master Content Factory | `CF_MINIO_ENDPOINT` | review | review_context | no | low |
| Master Content Factory | `CF_MINIO_PORT` | review | review_context | no | low |
| Master Content Factory | `CF_MINIO_SECRET_KEY` | review | review_context | yes | low |
| Master Content Factory | `CF_MINIO_USE_SSL` | review | review_context | no | low |
| Master Content Factory | `CF_N8N_API_KEY` | review | review_context | yes | low |
| Master Content Factory | `CF_N8N_WEBHOOK_URL` | review | review_context | no | low |
| Master Content Factory | `CF_OMISE_PUBLIC_KEY` | review | review_context | yes | low |
| Master Content Factory | `CF_OMISE_SECRET_KEY` | review | review_context | yes | low |
| Master Content Factory | `CF_REDIS_URL` | review | review_context | no | low |
| Master Content Factory | `CF_SUPABASE_ANON_KEY` | review | review_context | yes | low |
| Master Content Factory | `CF_SUPABASE_SERVICE_ROLE_KEY` | review | review_context | yes | low |
| Master Content Factory | `CF_SUPABASE_URL` | review | review_context | no | low |
| Master Content Factory | `CF_X_ACCESS_TOKEN` | review | review_context | yes | low |
| Master Content Factory | `CF_X_API_KEY` | review | review_context | yes | low |
| Master Content Factory | `CF_X_API_SECRET` | review | review_context | yes | low |
| Master Content Factory | `CF_YT_CLIENT_ID` | review | review_context | no | low |
| Master Content Factory | `CF_YT_CLIENT_SECRET` | review | review_context | yes | low |
| Master Content Factory | `CF_YT_REFRESH_TOKEN` | review | review_context | yes | low |
| Master Content Factory | `DEFAULT_STORAGE_QUOTA_MB` | review | review_context | no | low |
| Master Content Factory | `DIRECT_URL` | review | review_context | no | low |
| Master Content Factory | `DUCKDB_PATH` | review | review_context | no | low |
| Master Content Factory | `EXPORT_DIR` | review | review_context | no | low |
| Master Content Factory | `FINNHUB_API_KEY` | review | review_context | yes | low |
| Master Content Factory | `GITLAB_TOKEN` | review | review_context | yes | low |
| Master Content Factory | `GOOGLE_REDIRECT_URI` | review | review_context | no | low |
| Master Content Factory | `MAX_UPLOAD_SIZE_MB` | review | review_context | no | low |
| Master Content Factory | `NEXT_PUBLIC_APP_NAME` | review | review_context | no | low |
| Master Content Factory | `OPENCLAW_GATEWAY_PORT` | unused_env | review_remove_or_document | no | medium |
| Master Content Factory | `OPENROUTER_BASE_URL` | review | review_context | no | low |
| Master Content Factory | `RATE_LIMIT_MAX_REQUESTS` | review | review_context | no | low |
| Master Content Factory | `RATE_LIMIT_WINDOW_MS` | review | review_context | no | low |
| Master Content Factory | `REDIS_PASSWORD` | review | review_context | yes | low |
| Master Content Factory | `SENTRY_AUTH_TOKEN` | review | review_context | yes | low |
| Master Content Factory | `STRIPE_PUBLISHABLE_KEY` | review | review_context | yes | low |
| Master Content Factory | `TELEGRAM_BOT_TOKEN` | review | review_context | yes | low |
| Master Content Factory | `TWELVEDATA_API_KEY` | review | review_context | yes | low |
| Master Content Factory | `VPS_PASSWORD` | unused_env | review_remove_or_document | yes | medium |
| Master Content Factory | `VPS_SSH_KEY` | review | review_context | yes | low |
| Master Content Factory | `VPS_USER` | review | review_context | no | low |
| Master Fundamental | `AI_GATEWAY_DEFAULT_MODEL` | review | review_context | no | low |
| Master Fundamental | `GITLAB_TOKEN` | review | review_context | yes | low |
| Master Fundamental | `OPENCLAW_DEFAULT_MODEL` | unused_env | review_remove_or_document | no | medium |
| Master Fundamental | `OPENCLAW_GATEWAY_PORT` | unused_env | review_remove_or_document | no | medium |
| Master Fundamental | `OPENROUTER_API_KEY` | review | review_context | yes | low |
| Master Fundamental | `VPS_PASSWORD` | unused_env | review_remove_or_document | yes | medium |
| Master Fundamental | `VPS_SSH_KEY` | review | review_context | yes | low |
| Master Fundamental | `VPS_USER` | review | review_context | no | low |
| Master JigsawWebChat | `ADMIN_URL` | review | review_context | no | low |
| Master JigsawWebChat | `AI_GATEWAY_DEFAULT_MODEL` | review | review_context | no | low |
| Master JigsawWebChat | `API_URL` | review | review_context | no | low |
| Master JigsawWebChat | `GITLAB_TOKEN` | review | review_context | yes | low |
| Master JigsawWebChat | `OPENCLAW_DEFAULT_MODEL` | unused_env | review_remove_or_document | no | medium |
| Master JigsawWebChat | `OPENCLAW_GATEWAY_PORT` | unused_env | review_remove_or_document | no | medium |
| Master JigsawWebChat | `TESTSPRITE_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Master JigsawWebChat | `VPS_PASSWORD` | unused_env | review_remove_or_document | yes | medium |
| Master JigsawWebChat | `VPS_SSH_KEY` | review | review_context | yes | low |
| Master JigsawWebChat | `VPS_USER` | review | review_context | no | low |
| Master ScanlyIQ | `AUTH_GOOGLE_ID` | unused_env | review_remove_or_document | no | medium |
| Master ScanlyIQ | `AUTH_GOOGLE_SECRET` | unused_env | review_remove_or_document | yes | medium |
| Master ScanlyIQ | `AUTH_URL` | unused_env | review_remove_or_document | no | medium |
| Master ScanlyIQ | `BINANCE_API_KEY` | review | review_context | yes | low |
| Master ScanlyIQ | `BINANCE_SECRET_KEY` | review | review_context | yes | low |
| Master ScanlyIQ | `DATABASE_URL` | review | review_context | no | low |
| Master ScanlyIQ | `DEFAULT_STORAGE_QUOTA_MB` | unused_env | review_remove_or_document | no | medium |
| Master ScanlyIQ | `DIRECT_URL` | review | review_context | no | low |
| Master ScanlyIQ | `DUCKDB_PATH` | review | review_context | no | low |
| Master ScanlyIQ | `ELEVENLABS_WEBHOOK_SECRET` | review | review_context | yes | low |
| Master ScanlyIQ | `EXPORT_DIR` | unused_env | review_remove_or_document | no | medium |
| Master ScanlyIQ | `FINNHUB_API_KEY` | review | review_context | yes | low |
| Master ScanlyIQ | `GITLAB_TOKEN` | review | review_context | yes | low |
| Master ScanlyIQ | `GOOGLE_REDIRECT_URI` | unused_env | review_remove_or_document | no | medium |
| Master ScanlyIQ | `NEXT_PUBLIC_APP_NAME` | unused_env | review_remove_or_document | no | medium |
| Master ScanlyIQ | `OPENCLAW_DEFAULT_MODEL` | review | review_context | no | low |
| Master ScanlyIQ | `OPENCLAW_GATEWAY_PORT` | review | review_context | no | low |
| Master ScanlyIQ | `OPENROUTER_BASE_URL` | unused_env | review_remove_or_document | no | medium |
| Master ScanlyIQ | `RATE_LIMIT_MAX_REQUESTS` | unused_env | review_remove_or_document | no | medium |
| Master ScanlyIQ | `RATE_LIMIT_WINDOW_MS` | unused_env | review_remove_or_document | no | medium |
| Master ScanlyIQ | `STRIPE_PUBLISHABLE_KEY` | unused_env | review_remove_or_document | yes | medium |
| Master ScanlyIQ | `TELEGRAM_BOT_TOKEN` | review | review_context | yes | low |
| Master ScanlyIQ | `TEST_DATABASE_URL` | review | review_context | no | low |
| Master ScanlyIQ | `TWELVEDATA_API_KEY` | review | review_context | yes | low |
| Master ScanlyIQ | `VPS_SSH_KEY` | review | review_context | yes | low |
| Master ScanlyIQ | `VPS_USER` | review | review_context | no | low |
| Master ViberQC | `CRON_SECRET` | unused_env | review_remove_or_document | yes | medium |
| Master ViberQC | `DEPLOY_BRANCH` | unused_env | review_remove_or_document | no | medium |
| Master ViberQC | `GITLAB_USER` | unused_env | review_remove_or_document | no | medium |
| Master ViberQC | `LINUXNAT_PORT` | unused_env | review_remove_or_document | no | medium |
| Master ViberQC | `LINUXNAT_SUDO_PASSWORD` | unused_env | review_remove_or_document | yes | medium |
| Master ViberQC | `NEXTAUTH_SECRET` | review | review_context | yes | low |
| Master ViberQC | `NEXTAUTH_URL` | review | review_context | no | low |
| Master ViberQC | `NEXT_PUBLIC_APP_NAME` | review | review_context | no | low |
| Master ViberQC | `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY` | review | review_context | yes | low |
| Master ViberQC | `REDIS_URL` | review | review_context | no | low |
| Master ViberQC | `SENTRY_AUTH_TOKEN` | review | review_context | yes | low |
| Master ViberQC | `SONAR_TOKEN` | unused_env | review_remove_or_document | yes | medium |
| Master ViberQC | `SSH_CONFIG_NAME` | unused_env | review_remove_or_document | no | medium |
| Master ViberQC | `SSH_HOST` | unused_env | review_remove_or_document | no | medium |
| Master ViberQC | `SSH_KEY_PATH` | unused_env | review_remove_or_document | yes | medium |
| Master ViberQC | `SSH_PORT` | unused_env | review_remove_or_document | no | medium |
| Master ViberQC | `SSH_USER` | unused_env | review_remove_or_document | no | medium |
| Master ViberQC | `TESTSPRITE_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Master ViberQC | `VPS_PATH` | unused_env | review_remove_or_document | no | medium |
| Master_GodeysDB | `AI_DEFAULT_MODEL` | review | review_context | no | low |
| Master_GodeysDB | `AI_GATEWAY_PORT` | review | review_context | no | low |
| Master_GodeysDB | `DUCKDB_PATH` | review | review_context | no | low |
| Master_GodeysDB | `GITLAB_DEPLOY_TOKEN` | review | review_context | yes | low |
| Master_GodeysDB | `GITLAB_TOKEN` | review | review_context | yes | low |
| Master_GodeysDB | `LINE_CHANNEL_ACCESS_TOKEN` | review | review_context | yes | low |
| Master_GodeysDB | `LINE_CHANNEL_SECRET` | review | review_context | yes | low |
| Master_GodeysDB | `LINUXNAT_HOST` | review | review_context | no | low |
| Master_GodeysDB | `LINUXNAT_PORT` | review | review_context | no | low |
| Master_GodeysDB | `LINUXNAT_SSH_CONFIG_NAME` | review | review_context | no | low |
| Master_GodeysDB | `LINUXNAT_SUDO_PASSWORD` | review | review_context | yes | low |
| Master_GodeysDB | `LINUXNAT_USER` | review | review_context | no | low |
| Master_GodeysDB | `NETWORK_NAME` | review | review_context | no | low |
| Master_GodeysDB | `NEXT_PUBLIC_APP_NAME` | review | review_context | no | low |
| Master_GodeysDB | `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY` | review | review_context | yes | low |
| Master_GodeysDB | `OPENCLAW_DEFAULT_MODEL` | review | review_context | no | low |
| Master_GodeysDB | `PYTHON_ENGINE_PORT` | review | review_context | no | low |
| Master_GodeysDB | `REDIS_PORT` | review | review_context | no | low |
| Master_GodeysDB | `SENTRY_ORG` | review | review_context | no | low |
| Master_GodeysDB | `SENTRY_PROJECT` | review | review_context | no | low |
| Master_GodeysDB | `SONAR_TOKEN` | review | review_context | yes | low |
| Master_GodeysDB | `SSH_CONFIG_NAME` | review | review_context | no | low |
| Master_GodeysDB | `SSH_HOST` | review | review_context | no | low |
| Master_GodeysDB | `SSH_KEY_PATH` | review | review_context | yes | low |
| Master_GodeysDB | `SSH_PORT` | review | review_context | no | low |
| Master_GodeysDB | `SSH_USER` | review | review_context | no | low |
| Master_GodeysDB | `STRIPE_PUBLISHABLE_KEY` | review | review_context | yes | low |
| Master_GodeysDB | `TELEGRAM_BOT_TOKEN` | review | review_context | yes | low |
| Master_GodeysDB | `TESTSPRITE_API_KEY` | review | review_context | yes | low |
| Master_GodeysDB | `VPS_PASSWORD` | review | review_context | yes | low |
| SaaS Web Engine | `LINUXNAT_HOST` | review | review_context | no | low |
| SaaS Web Engine | `LINUXNAT_PORT` | review | review_context | no | low |
| SaaS Web Engine | `LINUXNAT_SSH_CONFIG_NAME` | review | review_context | no | low |
| SaaS Web Engine | `LINUXNAT_SUDO_PASSWORD` | review | review_context | yes | low |
| SaaS Web Engine | `LINUXNAT_USER` | review | review_context | no | low |
| Venture Radar | `ANTHROPIC_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Venture Radar | `ASANA_CLIENT_ID` | unused_env | review_remove_or_document | no | medium |
| Venture Radar | `ASANA_CLIENT_SECRET` | unused_env | review_remove_or_document | yes | medium |
| Venture Radar | `ASANA_PERSONAL_ACCESS_TOKEN` | unused_env | review_remove_or_document | yes | medium |
| Venture Radar | `DATABASE_URL` | review | review_context | no | low |
| Venture Radar | `DEEPSEEK_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Venture Radar | `GITLAB_TOKEN` | unused_env | review_remove_or_document | yes | medium |
| Venture Radar | `GITLAB_URL` | unused_env | review_remove_or_document | no | medium |
| Venture Radar | `LARK_BASE_APP_TOKEN` | unused_env | review_remove_or_document | yes | medium |
| Venture Radar | `LARK_DOMAIN` | unused_env | review_remove_or_document | no | medium |
| Venture Radar | `LARK_TABLE_CASHFLOW` | unused_env | review_remove_or_document | no | medium |
| Venture Radar | `LINUXNAT_HOST` | unused_env | review_remove_or_document | no | medium |
| Venture Radar | `LINUXNAT_PORT` | unused_env | review_remove_or_document | no | medium |
| Venture Radar | `LINUXNAT_SSH_CONFIG_NAME` | unused_env | review_remove_or_document | no | medium |
| Venture Radar | `LINUXNAT_SUDO_PASSWORD` | unused_env | review_remove_or_document | yes | medium |
| Venture Radar | `LINUXNAT_USER` | unused_env | review_remove_or_document | no | medium |
| Venture Radar | `MOONSHOT_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Venture Radar | `OPENAI_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Venture Radar | `VPS_SSH_KEY` | unused_env | review_remove_or_document | yes | medium |
| Venture Radar | `VPS_SSH_PORT` | unused_env | review_remove_or_document | no | medium |
| Venture Radar | `WEB_PORT` | review | review_context | no | low |
| Venture Radar-Bak | `COS_AGENT_ID` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `DAILY_TRIGGER_TIME` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `DASHBOARD_URL` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `DASHBOARD_VPS_PATH` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `FINANCE_AGENT_ID` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `GTM_AGENT_ID` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `LARK_BASE_TBL_RULES` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `LARK_BASE_URL` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `LARK_ENCRYPT_KEY` | unused_env | review_remove_or_document | yes | medium |
| Venture Radar-Bak | `LARK_NOTIFY_WEBHOOK` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `LARK_RECEIVE_ID_TYPE` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `LARK_VERIFICATION_TOKEN` | unused_env | review_remove_or_document | yes | medium |
| Venture Radar-Bak | `LOG_LEVEL` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `N8N_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Venture Radar-Bak | `N8N_URL` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `NAT_PERSONAL_CHAT_ID` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `OPENCLAW_URL` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `OPENROUTER_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| Venture Radar-Bak | `PAPERCLIP_URL` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `SCOUT_AGENT_ID` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `TZ` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `VENTURE_RADAR_COMPANY_ID` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `VENTURE_RADAR_PREFIX` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `VPS_SSH_PORT` | unused_env | review_remove_or_document | no | medium |
| Venture Radar-Bak | `WEEKLY_DIGEST_TIME` | unused_env | review_remove_or_document | no | medium |
| claude | `NEXTAUTH_SECRET` | review | review_context | yes | low |
| claude | `NEXTAUTH_URL` | review | review_context | no | low |
| claude | `NEXT_PUBLIC_APP_NAME` | review | review_context | no | low |
| claude | `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY` | review | review_context | yes | low |
| claude | `SENTRY_AUTH_TOKEN` | review | review_context | yes | low |
| codex | `NEXTAUTH_SECRET` | review | review_context | yes | low |
| codex | `NEXTAUTH_URL` | review | review_context | no | low |
| codex | `NEXT_PUBLIC_APP_NAME` | review | review_context | no | low |
| codex | `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY` | review | review_context | yes | low |
| codex | `SENTRY_AUTH_TOKEN` | review | review_context | yes | low |
| gemini | `NEXTAUTH_SECRET` | review | review_context | yes | low |
| gemini | `NEXTAUTH_URL` | review | review_context | no | low |
| gemini | `NEXT_PUBLIC_APP_NAME` | review | review_context | no | low |
| gemini | `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY` | review | review_context | yes | low |
| gemini | `SENTRY_AUTH_TOKEN` | review | review_context | yes | low |
| qwen | `NEXTAUTH_SECRET` | review | review_context | yes | low |
| qwen | `NEXTAUTH_URL` | review | review_context | no | low |
| qwen | `NEXT_PUBLIC_APP_NAME` | review | review_context | no | low |
| qwen | `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY` | review | review_context | yes | low |
| qwen | `SENTRY_AUTH_TOKEN` | review | review_context | yes | low |
| AI In Office | `LINUXNAT_HOST` | review | review_context | no | low |
| AI In Office | `LINUXNAT_PORT` | review | review_context | no | low |
| AI In Office | `LINUXNAT_SSH_CONFIG_NAME` | review | review_context | no | low |
| AI In Office | `LINUXNAT_SUDO_PASSWORD` | unused_env | review_remove_or_document | yes | medium |
| AI In Office | `LINUXNAT_USER` | review | review_context | no | low |
| AI In Office | `POSTGRES_PASSWORD` | review | review_context | yes | low |
| AI on Premis | `AIRGAP` | review | review_context | no | low |
| AI on Premis | `GROK_API_KEY` | review | review_context | yes | low |
| AI on Premis | `LOG_LEVEL` | review | review_context | no | low |
| AIControlCenter | `LINUXNAT_SSH_CONFIG_NAME` | review | review_context | no | low |
| AIControlCenter | `POSTGRES_PASSWORD` | review | review_context | yes | low |
| Dev Ser | `GRAFANA_BASE_URL` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `GRAFANA_COMPARE_ALL_DASHBOARD` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `GRAFANA_DISK_PER_PROJECT_DASHBOARD` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `LINUXNAT_HOST` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `LINUXNAT_PORT` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `LINUXNAT_SSH_CONFIG_NAME` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `LINUXNAT_SUDO_PASSWORD` | unused_env | review_remove_or_document | yes | medium |
| Dev Ser | `LINUXNAT_USER` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `LXC_MONITORING_ID` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `LXC_WAZUH_ID` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `PROXMOX_KERNEL` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `PROXMOX_MAX_CPU` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `PROXMOX_MAX_MEM_GIB` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `PROXMOX_NODE_NAME` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `PROXMOX_STORAGE_BACKUP` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `PROXMOX_STORAGE_LOCAL_ZFS` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `PROXMOX_VERSION` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `SSH_CONFIG_NAME` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `SSH_HOST` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `SSH_KEY_PATH` | unused_env | review_remove_or_document | yes | medium |
| Dev Ser | `SSH_PORT` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `SSH_USER` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `VM_GITLAB_ID` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `VM_LINUX_DEV_ID` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `VM_LINUX_NAT_ID` | unused_env | review_remove_or_document | no | medium |
| Dev Ser | `VM_WINSERVER_ID` | unused_env | review_remove_or_document | no | medium |
| EmailHunter | `LINUXNAT_HOST` | unused_env | review_remove_or_document | no | medium |
| EmailHunter | `LINUXNAT_PORT` | unused_env | review_remove_or_document | no | medium |
| EmailHunter | `LINUXNAT_SSH_CONFIG_NAME` | unused_env | review_remove_or_document | no | medium |
| EmailHunter | `LINUXNAT_SUDO_PASSWORD` | unused_env | review_remove_or_document | yes | medium |
| EmailHunter | `LINUXNAT_USER` | unused_env | review_remove_or_document | no | medium |
| EmailHunter | `N8N_ENCRYPTION_KEY` | review | review_context | yes | low |
| Hermes Agent | `BROWSER_SESSION_TIMEOUT` | review | review_context | no | low |
| Hermes Agent | `IMAGE_TOOLS_DEBUG` | review | review_context | no | low |
| Hermes Agent | `MOA_TOOLS_DEBUG` | review | review_context | no | low |
| Hermes Agent | `VISION_TOOLS_DEBUG` | review | review_context | no | low |
| Hermes Agent | `WEB_TOOLS_DEBUG` | review | review_context | no | low |
| Index to VPS | `GITLAB_APP_ID` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `GITLAB_APP_SECRET` | unused_env | review_remove_or_document | yes | medium |
| Index to VPS | `GITLAB_BASE_URL` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `GRAFANA_BASE_URL` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `GRAFANA_COMPARE_ALL_DASHBOARD` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `GRAFANA_DISK_PER_PROJECT_DASHBOARD` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `LINUXNAT_HOST` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `LINUXNAT_PORT` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `LINUXNAT_SSH_CONFIG_NAME` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `LINUXNAT_SUDO_PASSWORD` | unused_env | review_remove_or_document | yes | medium |
| Index to VPS | `LINUXNAT_USER` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `LXC_MONITORING_ID` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `LXC_WAZUH_ID` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `PROXMOX_KERNEL` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `PROXMOX_MAX_CPU` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `PROXMOX_MAX_MEM_GIB` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `PROXMOX_NODE_NAME` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `PROXMOX_STORAGE_BACKUP` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `PROXMOX_STORAGE_LOCAL_ZFS` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `PROXMOX_VERSION` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `SSH_CONFIG_NAME` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `SSH_HOST` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `SSH_KEY_PATH` | unused_env | review_remove_or_document | yes | medium |
| Index to VPS | `SSH_PORT` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `SSH_USER` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `VM_GITLAB_ID` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `VM_LINUX_DEV_ID` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `VM_LINUX_NAT_ID` | unused_env | review_remove_or_document | no | medium |
| Index to VPS | `VM_WINSERVER_ID` | unused_env | review_remove_or_document | no | medium |
| Main Server | `GRAFANA_BASE_URL` | unused_env | review_remove_or_document | no | medium |
| Main Server | `GRAFANA_COMPARE_ALL_DASHBOARD` | unused_env | review_remove_or_document | no | medium |
| Main Server | `GRAFANA_DISK_PER_PROJECT_DASHBOARD` | unused_env | review_remove_or_document | no | medium |
| Main Server | `LINUXNAT_HOST` | unused_env | review_remove_or_document | no | medium |
| Main Server | `LINUXNAT_PORT` | unused_env | review_remove_or_document | no | medium |
| Main Server | `LINUXNAT_SSH_CONFIG_NAME` | unused_env | review_remove_or_document | no | medium |
| Main Server | `LINUXNAT_SUDO_PASSWORD` | unused_env | review_remove_or_document | yes | medium |
| Main Server | `LINUXNAT_USER` | unused_env | review_remove_or_document | no | medium |
| Main Server | `LXC_MONITORING_ID` | unused_env | review_remove_or_document | no | medium |
| Main Server | `LXC_WAZUH_ID` | unused_env | review_remove_or_document | no | medium |
| Main Server | `PROXMOX_KERNEL` | unused_env | review_remove_or_document | no | medium |
| Main Server | `PROXMOX_MAX_CPU` | unused_env | review_remove_or_document | no | medium |
| Main Server | `PROXMOX_MAX_MEM_GIB` | unused_env | review_remove_or_document | no | medium |
| Main Server | `PROXMOX_NODE_NAME` | unused_env | review_remove_or_document | no | medium |
| Main Server | `PROXMOX_STORAGE_BACKUP` | unused_env | review_remove_or_document | no | medium |
| Main Server | `PROXMOX_STORAGE_LOCAL_ZFS` | unused_env | review_remove_or_document | no | medium |
| Main Server | `PROXMOX_VERSION` | unused_env | review_remove_or_document | no | medium |
| Main Server | `SSH_CONFIG_NAME` | unused_env | review_remove_or_document | no | medium |
| Main Server | `SSH_HOST` | unused_env | review_remove_or_document | no | medium |
| Main Server | `SSH_KEY_PATH` | unused_env | review_remove_or_document | yes | medium |
| Main Server | `SSH_PORT` | unused_env | review_remove_or_document | no | medium |
| Main Server | `SSH_USER` | unused_env | review_remove_or_document | no | medium |
| Main Server | `VM_GITLAB_ID` | unused_env | review_remove_or_document | no | medium |
| Main Server | `VM_LINUX_DEV_ID` | unused_env | review_remove_or_document | no | medium |
| Main Server | `VM_LINUX_NAT_ID` | unused_env | review_remove_or_document | no | medium |
| Main Server | `VM_WINSERVER_ID` | unused_env | review_remove_or_document | no | medium |
| OpenClaw | `LINUXNAT_HOST` | unused_env | review_remove_or_document | no | medium |
| OpenClaw | `LINUXNAT_PORT` | unused_env | review_remove_or_document | no | medium |
| OpenClaw | `LINUXNAT_SSH_CONFIG_NAME` | unused_env | review_remove_or_document | no | medium |
| OpenClaw | `LINUXNAT_SUDO_PASSWORD` | unused_env | review_remove_or_document | yes | medium |
| OpenClaw | `LINUXNAT_USER` | unused_env | review_remove_or_document | no | medium |
| OpenClaw | `SSH_CONFIG_NAME` | unused_env | review_remove_or_document | no | medium |
| OpenClaw | `SSH_HOST` | unused_env | review_remove_or_document | no | medium |
| OpenClaw | `SSH_KEY_PATH` | unused_env | review_remove_or_document | yes | medium |
| OpenClaw | `SSH_PORT` | unused_env | review_remove_or_document | no | medium |
| OpenClaw | `SSH_USER` | unused_env | review_remove_or_document | no | medium |
| OpenClaw | `SUDO_PASS` | unused_env | review_remove_or_document | no | medium |
| OpenClaw-EGP | `ADMIN_API_TOKEN` | review | review_context | yes | low |
| OpenClaw-EGP | `BASE_URL` | review | review_context | no | low |
| OpenClaw-EGP | `DATABASE_PATH` | review | review_context | no | low |
| OpenClaw-EGP | `EGP_MAX_PAGES_PER_SCAN` | review | review_context | no | low |
| OpenClaw-EGP | `EGP_MIN_BUDGET` | review | review_context | no | low |
| OpenClaw-EGP | `EGP_SCAN_CRON` | review | review_context | no | low |
| OpenClaw-EGP | `EGP_SCAN_ENABLED` | review | review_context | no | low |
| OpenClaw-EGP | `EGP_SCAN_MAX_RETRIES` | review | review_context | no | low |
| OpenClaw-EGP | `EGP_SOURCE_URL` | review | review_context | no | low |
| OpenClaw-EGP | `LARK_ENCRYPT_KEY` | review | review_context | yes | low |
| OpenClaw-EGP | `LARK_SEND_ENABLED` | review | review_context | no | low |
| OpenClaw-EGP | `LARK_VERIFICATION_TOKEN` | review | review_context | yes | low |
| OpenClaw-EGP | `LARK_WEBHOOK_URL` | review | review_context | no | low |
| OpenClaw-EGP | `LOG_LEVEL` | review | review_context | no | low |
| OpenClaw-EGP | `NODE_ENV` | review | review_context | no | low |
| VPS Server | `COCKPIT_PORT` | review | review_context | no | low |
| VPS Server | `SERVER_DOMAIN` | review | review_context | no | low |
| VPS Server | `SERVER_NAME` | review | review_context | no | low |
| Hermes Labs.retired-20260524 | `ANTHROPIC_API_KEY` | unused_env | review_remove_or_document | yes | medium |
| HermesNous.retired-20260524 | `LINUXNAT_HOST` | unused_env | review_remove_or_document | no | medium |
| HermesNous.retired-20260524 | `LINUXNAT_PORT` | unused_env | review_remove_or_document | no | medium |
| HermesNous.retired-20260524 | `LINUXNAT_SSH_CONFIG_NAME` | unused_env | review_remove_or_document | no | medium |
| HermesNous.retired-20260524 | `LINUXNAT_SUDO_PASSWORD` | unused_env | review_remove_or_document | yes | medium |
| HermesNous.retired-20260524 | `LINUXNAT_USER` | unused_env | review_remove_or_document | no | medium |
