-- Allow runtime replacement of quote line items while keeping broader CRM deletes unavailable.
GRANT DELETE ON crm.quote_items TO crm_runtime;
