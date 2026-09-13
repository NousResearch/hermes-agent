PRAGMA foreign_keys=ON;

CREATE TABLE kanban_board_identity (
  singleton INTEGER PRIMARY KEY CHECK (singleton=1),
  board_instance_id TEXT NOT NULL UNIQUE CHECK (length(board_instance_id) BETWEEN 16 AND 128),
  canonical_board_key TEXT NOT NULL CHECK (length(canonical_board_key) BETWEEN 1 AND 256),
  created_at INTEGER NOT NULL CHECK (created_at>0)
);

CREATE TABLE kanban_schema_migrations (
  migration_id TEXT PRIMARY KEY,
  target_schema_version INTEGER NOT NULL UNIQUE CHECK (target_schema_version>0),
  state TEXT NOT NULL CHECK (state IN ('prepared','applied','verified','failed')),
  board_instance_id TEXT NOT NULL,
  source_digest TEXT NOT NULL CHECK (length(source_digest)=64),
  backup_digest TEXT NOT NULL CHECK (length(backup_digest)=64),
  receipt_digest TEXT,
  fence_generation INTEGER NOT NULL CHECK (fence_generation>=0),
  prepared_at INTEGER NOT NULL,
  applied_at INTEGER,
  verified_at INTEGER,
  FOREIGN KEY (board_instance_id) REFERENCES kanban_board_identity(board_instance_id),
  CHECK ((state='verified' AND receipt_digest IS NOT NULL AND length(receipt_digest)=64 AND verified_at IS NOT NULL)
      OR (state!='verified'))
);

CREATE TABLE kanban_write_fence (
  singleton INTEGER PRIMARY KEY CHECK (singleton=1),
  mode TEXT NOT NULL CHECK (mode IN ('open','draining','fenced')),
  generation INTEGER NOT NULL CHECK (generation>=0),
  owner_token_hash TEXT,
  reason TEXT,
  updated_at INTEGER NOT NULL,
  CHECK ((mode='open' AND owner_token_hash IS NULL) OR (mode!='open' AND owner_token_hash IS NOT NULL))
);

CREATE TABLE kanban_commit_clock (
  singleton INTEGER PRIMARY KEY CHECK (singleton=1),
  commit_seq INTEGER NOT NULL CHECK (commit_seq>=0),
  last_txn_id TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE kanban_migration_operations (
  migration_id TEXT PRIMARY KEY,
  board_instance_id TEXT NOT NULL,
  owner_token_hash TEXT NOT NULL CHECK (length(owner_token_hash)=64),
  source_digest TEXT NOT NULL CHECK (length(source_digest)=64),
  plan_digest TEXT NOT NULL CHECK (length(plan_digest)=64),
  schema_digest TEXT NOT NULL CHECK (length(schema_digest)=64),
  backup_digest TEXT,
  fence_generation INTEGER NOT NULL CHECK (fence_generation>=0),
  phase TEXT NOT NULL CHECK (phase IN (
    'prepared','barrier_held','backup_sealed','committed',
    'receipt_completed','verified','unfenced','failed'
  )),
  phase_revision INTEGER NOT NULL CHECK (phase_revision>=0),
  receipt_digest TEXT,
  started_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  FOREIGN KEY (board_instance_id) REFERENCES kanban_board_identity(board_instance_id)
);

CREATE TABLE orch_rollback_operations (
  rollback_id TEXT PRIMARY KEY,
  migration_id TEXT NOT NULL,
  board_instance_id TEXT NOT NULL,
  owner_token_hash TEXT NOT NULL CHECK (length(owner_token_hash)=64),
  source_manifest_digest TEXT NOT NULL CHECK (length(source_manifest_digest)=64),
  target_manifest_digest TEXT NOT NULL CHECK (length(target_manifest_digest)=64),
  fence_generation INTEGER NOT NULL CHECK (fence_generation>=0),
  phase TEXT NOT NULL CHECK (phase IN (
    'gate_submits','fence_draining','workers_stopped','leases_revoked',
    'snapshot_sealed','code_switched','old_writer_receipts_verified','verified','reopened','failed'
  )),
  phase_revision INTEGER NOT NULL CHECK (phase_revision>=0),
  snapshot_digest TEXT,
  started_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  FOREIGN KEY (migration_id) REFERENCES kanban_migration_operations(migration_id),
  FOREIGN KEY (board_instance_id) REFERENCES kanban_board_identity(board_instance_id)
);

CREATE TABLE orch_replay_selectors (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  selector_key TEXT NOT NULL CHECK (length(selector_key)=64),
  selector_kind TEXT NOT NULL CHECK (selector_kind IN ('event','client')),
  selector_value TEXT NOT NULL CHECK (length(selector_value)>0),
  adapter_instance_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  lineage_id TEXT NOT NULL,
  current_generation INTEGER NOT NULL CHECK (current_generation>=0),
  current_orch_id TEXT,
  current_request_digest TEXT,
  ledger_revision INTEGER NOT NULL CHECK (ledger_revision>=0),
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (board_instance_id,tenant_scope,selector_key),
  UNIQUE (board_instance_id,tenant_scope,adapter_instance_id,conversation_id,selector_kind,selector_value),
  UNIQUE (board_instance_id,tenant_scope,lineage_id),
  FOREIGN KEY (board_instance_id) REFERENCES kanban_board_identity(board_instance_id),
  CHECK ((current_generation=0 AND current_orch_id IS NULL AND current_request_digest IS NULL)
      OR (current_generation>0 AND current_orch_id IS NOT NULL AND length(current_request_digest)=64))
);

CREATE TABLE orch_origins (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  origin_id TEXT NOT NULL,
  schema_version INTEGER NOT NULL CHECK (schema_version=4),
  selector_key TEXT NOT NULL CHECK (length(selector_key)=64),
  origin_kind TEXT NOT NULL CHECK (origin_kind IN ('messaging','api','local_session','board_only')),
  platform TEXT NOT NULL,
  adapter_instance_id TEXT NOT NULL,
  account_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  selector_kind TEXT NOT NULL CHECK (selector_kind IN ('event','client')),
  selector_value TEXT NOT NULL CHECK (length(selector_value)>0),
  client_idempotency_key TEXT,
  thread_id TEXT NOT NULL,
  reply_to_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  notifier_profile TEXT NOT NULL,
  route_revision INTEGER NOT NULL CHECK (route_revision>0),
  route_json TEXT NOT NULL,
  route_digest TEXT NOT NULL CHECK (length(route_digest)=64),
  required_ack_family TEXT NOT NULL CHECK (required_ack_family IN (
    'provider','adapter','synchronous','local_session','none'
  )),
  required_ack_strength TEXT NOT NULL CHECK (required_ack_strength IN (
    'provider_message_id','adapter_acceptance','synchronous_response',
    'local_session_acceptance','none'
  )),
  created_at INTEGER NOT NULL,
  PRIMARY KEY (board_instance_id,tenant_scope,origin_id),
  UNIQUE (board_instance_id,tenant_scope,origin_id,route_revision,route_digest),
  UNIQUE (board_instance_id,tenant_scope,selector_key,route_revision),
  FOREIGN KEY (board_instance_id) REFERENCES kanban_board_identity(board_instance_id),
  FOREIGN KEY (board_instance_id,tenant_scope,selector_key)
    REFERENCES orch_replay_selectors(board_instance_id,tenant_scope,selector_key),
  CHECK ((selector_kind='client' AND client_idempotency_key=selector_value)
      OR selector_kind='event'),
  CHECK ((origin_kind='board_only' AND required_ack_strength='none') OR
         (origin_kind!='board_only' AND required_ack_strength!='none')),
  CHECK ((required_ack_family='provider' AND required_ack_strength='provider_message_id')
      OR (required_ack_family='adapter' AND required_ack_strength='adapter_acceptance')
      OR (required_ack_family='synchronous' AND required_ack_strength='synchronous_response')
      OR (required_ack_family='local_session' AND required_ack_strength='local_session_acceptance')
      OR (required_ack_family='none' AND required_ack_strength='none'))
);

CREATE TABLE orch_requests (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  orch_id TEXT NOT NULL,
  lineage_id TEXT NOT NULL,
  generation INTEGER NOT NULL CHECK (generation>0),
  selector_key TEXT NOT NULL CHECK (length(selector_key)=64),
  selector_ledger_revision INTEGER NOT NULL CHECK (selector_ledger_revision>=0),
  request_key TEXT NOT NULL CHECK (length(request_key)=64),
  request_schema_version INTEGER NOT NULL CHECK (request_schema_version=4),
  request_json TEXT NOT NULL,
  request_digest TEXT NOT NULL CHECK (length(request_digest)=64),
  origin_id TEXT NOT NULL,
  parent_task_id TEXT NOT NULL,
  lifecycle_state TEXT NOT NULL CHECK (lifecycle_state IN (
    'submitted','decomposing','waiting_lanes','synthesizing','work_accepted',
    'delivering','delivery_blocked','blocked','cancelling','completed','failed','cancelled'
  )),
  lifecycle_revision INTEGER NOT NULL DEFAULT 0 CHECK (lifecycle_revision>=0),
  cancel_epoch INTEGER NOT NULL DEFAULT 0 CHECK (cancel_epoch>=0),
  delivery_epoch_revision INTEGER NOT NULL DEFAULT 0 CHECK (
    delivery_epoch_revision>=0 AND delivery_epoch_revision<=lifecycle_revision
  ),
  plan_epoch_revision INTEGER NOT NULL DEFAULT 0 CHECK (
    plan_epoch_revision>=0 AND plan_epoch_revision<=lifecycle_revision
  ),
  plan_version INTEGER NOT NULL DEFAULT 0 CHECK (plan_version>=0),
  synthesis_strategy TEXT NOT NULL CHECK (synthesis_strategy IN ('parent_owned','separate_node')),
  blocked_from_state TEXT,
  resume_state TEXT,
  block_kind TEXT,
  block_revision INTEGER NOT NULL DEFAULT 0 CHECK (block_revision>=0),
  input_request_id INTEGER,
  retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count>=0),
  max_retries INTEGER NOT NULL CHECK (max_retries BETWEEN 0 AND 100),
  deadline_at INTEGER,
  work_accepted_at INTEGER,
  delivery_closed_at INTEGER,
  terminal_reason_code TEXT,
  supersedes_orch_id TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (board_instance_id,tenant_scope,orch_id),
  UNIQUE (board_instance_id,tenant_scope,lineage_id,generation),
  UNIQUE (board_instance_id,tenant_scope,request_key),
  UNIQUE (board_instance_id,tenant_scope,parent_task_id),
  UNIQUE (board_instance_id,tenant_scope,supersedes_orch_id),
  UNIQUE (board_instance_id,tenant_scope,orch_id,request_key,request_digest,origin_id,parent_task_id),
  UNIQUE (board_instance_id,tenant_scope,orch_id,lineage_id,generation,request_key,
          request_digest,origin_id,parent_task_id,synthesis_strategy),
  FOREIGN KEY (board_instance_id,tenant_scope,selector_key)
    REFERENCES orch_replay_selectors(board_instance_id,tenant_scope,selector_key),
  FOREIGN KEY (board_instance_id,tenant_scope,origin_id)
    REFERENCES orch_origins(board_instance_id,tenant_scope,origin_id),
  FOREIGN KEY (parent_task_id) REFERENCES tasks(id) DEFERRABLE INITIALLY DEFERRED,
  FOREIGN KEY (board_instance_id,tenant_scope,supersedes_orch_id)
    REFERENCES orch_requests(board_instance_id,tenant_scope,orch_id),
  CHECK (
    (lifecycle_state='blocked' AND blocked_from_state IN ('decomposing','waiting_lanes','synthesizing')
       AND resume_state=blocked_from_state AND block_kind IN ('needs_input','capability','policy','external'))
    OR
    (lifecycle_state!='blocked' AND blocked_from_state IS NULL AND resume_state IS NULL AND block_kind IS NULL)
  ),
  CHECK ((lifecycle_state IN ('completed','failed','cancelled') AND terminal_reason_code IS NOT NULL)
      OR (lifecycle_state NOT IN ('completed','failed','cancelled'))),
  CHECK ((lifecycle_state='completed' AND work_accepted_at IS NOT NULL AND delivery_closed_at IS NOT NULL)
      OR lifecycle_state!='completed')
);

CREATE TABLE orch_request_requirements (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  orch_id TEXT NOT NULL,
  requirement_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL CHECK (ordinal>0),
  requirement_json TEXT NOT NULL,
  requirement_digest TEXT NOT NULL CHECK (length(requirement_digest)=64),
  required INTEGER NOT NULL CHECK (required IN (0,1)),
  PRIMARY KEY (board_instance_id,tenant_scope,orch_id,requirement_id),
  UNIQUE (board_instance_id,tenant_scope,orch_id,ordinal),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id)
    REFERENCES orch_requests(board_instance_id,tenant_scope,orch_id)
);

CREATE TABLE orch_plans (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  orch_id TEXT NOT NULL,
  plan_version INTEGER NOT NULL CHECK (plan_version>0),
  schema_version INTEGER NOT NULL CHECK (schema_version=4),
  lineage_id TEXT NOT NULL,
  generation INTEGER NOT NULL,
  request_key TEXT NOT NULL,
  request_digest TEXT NOT NULL,
  origin_id TEXT NOT NULL,
  parent_task_id TEXT NOT NULL,
  synthesis_strategy TEXT NOT NULL CHECK (synthesis_strategy IN ('parent_owned','separate_node')),
  plan_json TEXT NOT NULL,
  plan_digest TEXT NOT NULL CHECK (length(plan_digest)=64),
  created_by_run_id INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  PRIMARY KEY (board_instance_id,tenant_scope,orch_id,plan_version),
  UNIQUE (board_instance_id,tenant_scope,orch_id,plan_digest),
  UNIQUE (board_instance_id,tenant_scope,orch_id,plan_version,plan_digest),
  UNIQUE (board_instance_id,tenant_scope,orch_id,plan_version,request_digest,plan_digest),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id,lineage_id,generation,request_key,
               request_digest,origin_id,parent_task_id,synthesis_strategy)
    REFERENCES orch_requests(board_instance_id,tenant_scope,orch_id,lineage_id,generation,request_key,
                             request_digest,origin_id,parent_task_id,synthesis_strategy),
  CHECK (generation>0)
);

CREATE TABLE orch_plan_nodes (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  orch_id TEXT NOT NULL,
  plan_version INTEGER NOT NULL,
  node_key TEXT NOT NULL,
  lane_lineage_key TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('parent','lane','synthesis')),
  lane_label TEXT NOT NULL,
  normalized_goal TEXT NOT NULL,
  normalized_done_when TEXT NOT NULL,
  required INTEGER NOT NULL CHECK (required IN (0,1)),
  route_json TEXT NOT NULL,
  route_digest TEXT NOT NULL CHECK (length(route_digest)=64),
  ordinal INTEGER NOT NULL CHECK (ordinal>0),
  PRIMARY KEY (board_instance_id,tenant_scope,orch_id,plan_version,node_key),
  UNIQUE (board_instance_id,tenant_scope,orch_id,plan_version,ordinal),
  UNIQUE (board_instance_id,tenant_scope,orch_id,plan_version,lane_lineage_key),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id,plan_version)
    REFERENCES orch_plans(board_instance_id,tenant_scope,orch_id,plan_version),
  CHECK ((role='lane' AND lane_label!='') OR (role!='lane' AND lane_label='')),
  CHECK ((role='lane') OR required=1)
);

CREATE TABLE orch_plan_edges (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  orch_id TEXT NOT NULL,
  plan_version INTEGER NOT NULL,
  edge_key TEXT NOT NULL,
  parent_node_key TEXT NOT NULL,
  child_node_key TEXT NOT NULL,
  edge_kind TEXT NOT NULL CHECK (edge_kind IN ('orch_required_for_synthesis','orch_optional_context')),
  PRIMARY KEY (board_instance_id,tenant_scope,orch_id,plan_version,edge_key),
  UNIQUE (board_instance_id,tenant_scope,orch_id,plan_version,parent_node_key,child_node_key,edge_kind),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id,plan_version,parent_node_key)
    REFERENCES orch_plan_nodes(board_instance_id,tenant_scope,orch_id,plan_version,node_key),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id,plan_version,child_node_key)
    REFERENCES orch_plan_nodes(board_instance_id,tenant_scope,orch_id,plan_version,node_key),
  CHECK (parent_node_key!=child_node_key)
);

CREATE TABLE orch_plan_coverage (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  orch_id TEXT NOT NULL,
  plan_version INTEGER NOT NULL,
  requirement_id TEXT NOT NULL,
  node_key TEXT NOT NULL,
  PRIMARY KEY (board_instance_id,tenant_scope,orch_id,plan_version,requirement_id,node_key),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id,requirement_id)
    REFERENCES orch_request_requirements(board_instance_id,tenant_scope,orch_id,requirement_id),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id,plan_version,node_key)
    REFERENCES orch_plan_nodes(board_instance_id,tenant_scope,orch_id,plan_version,node_key)
);

CREATE TABLE orch_plan_materializations (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  orch_id TEXT NOT NULL,
  plan_version INTEGER NOT NULL,
  request_lifecycle_revision INTEGER NOT NULL CHECK (request_lifecycle_revision>=0),
  cancel_epoch INTEGER NOT NULL CHECK (cancel_epoch>=0),
  lease_epoch INTEGER NOT NULL CHECK (lease_epoch>0),
  plan_digest TEXT NOT NULL,
  observed_graph_digest TEXT NOT NULL CHECK (length(observed_graph_digest)=64),
  observed_node_count INTEGER NOT NULL CHECK (observed_node_count>=3),
  observed_edge_count INTEGER NOT NULL CHECK (observed_edge_count>=2),
  materialized_by_run_id INTEGER NOT NULL,
  commit_seq INTEGER NOT NULL CHECK (commit_seq>0),
  materialized_at INTEGER NOT NULL,
  PRIMARY KEY (board_instance_id,tenant_scope,orch_id,plan_version),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id,plan_version,plan_digest)
    REFERENCES orch_plans(board_instance_id,tenant_scope,orch_id,plan_version,plan_digest)
);

CREATE TABLE orch_nodes (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  orch_id TEXT NOT NULL,
  plan_version INTEGER NOT NULL,
  node_key TEXT NOT NULL,
  task_id TEXT NOT NULL,
  node_state TEXT NOT NULL CHECK (node_state IN ('planned','ready','running','cancellation_requested','accepted','blocked','failed','cancelled')),
  current_route_digest TEXT NOT NULL,
  route_revision INTEGER NOT NULL CHECK (route_revision>0),
  created_by_run_id INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (board_instance_id,tenant_scope,orch_id,node_key),
  UNIQUE (board_instance_id,tenant_scope,task_id),
  UNIQUE (board_instance_id,tenant_scope,orch_id,plan_version,node_key,task_id),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id,plan_version,node_key)
    REFERENCES orch_plan_nodes(board_instance_id,tenant_scope,orch_id,plan_version,node_key),
  FOREIGN KEY (task_id) REFERENCES tasks(id) DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE orch_external_edges (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  orch_id TEXT NOT NULL,
  edge_key TEXT NOT NULL,
  parent_task_id TEXT NOT NULL,
  child_task_id TEXT NOT NULL,
  edge_kind TEXT NOT NULL CHECK (edge_kind='orch_work_accepted'),
  lifecycle_revision INTEGER NOT NULL,
  created_by_run_id INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  PRIMARY KEY (board_instance_id,tenant_scope,orch_id,edge_key),
  UNIQUE (board_instance_id,tenant_scope,parent_task_id,child_task_id),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id)
    REFERENCES orch_requests(board_instance_id,tenant_scope,orch_id),
  FOREIGN KEY (parent_task_id) REFERENCES tasks(id),
  FOREIGN KEY (child_task_id) REFERENCES tasks(id)
);

CREATE TABLE orch_node_acceptances (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  orch_id TEXT NOT NULL,
  plan_version INTEGER NOT NULL,
  node_key TEXT NOT NULL,
  task_id TEXT NOT NULL,
  accepted_run_id INTEGER NOT NULL,
  outcome_digest TEXT NOT NULL CHECK (length(outcome_digest)=64),
  accepted_by_run_id INTEGER NOT NULL,
  acceptance_lease_epoch INTEGER NOT NULL CHECK (acceptance_lease_epoch>0),
  plan_epoch_revision INTEGER NOT NULL,
  cancel_epoch INTEGER NOT NULL,
  accepted_at INTEGER NOT NULL,
  PRIMARY KEY (board_instance_id,tenant_scope,orch_id,node_key),
  UNIQUE (board_instance_id,tenant_scope,accepted_run_id),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id,plan_version,node_key,task_id)
    REFERENCES orch_nodes(board_instance_id,tenant_scope,orch_id,plan_version,node_key,task_id),
  FOREIGN KEY (accepted_run_id) REFERENCES task_runs(id),
  FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE TABLE orch_stage_leases (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  orch_id TEXT NOT NULL,
  stage TEXT NOT NULL CHECK (stage IN ('decomposition','synthesis','reconciliation')),
  owner_run_id INTEGER NOT NULL,
  owner_profile TEXT NOT NULL,
  token_hash TEXT NOT NULL CHECK (length(token_hash)=64),
  epoch INTEGER NOT NULL CHECK (epoch>0),
  expires_at INTEGER NOT NULL,
  lease_state TEXT NOT NULL CHECK (lease_state IN ('active','released','revoked')),
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (board_instance_id,tenant_scope,orch_id,stage),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id)
    REFERENCES orch_requests(board_instance_id,tenant_scope,orch_id)
);

CREATE TABLE orch_stage_attempts (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  orch_id TEXT NOT NULL,
  stage TEXT NOT NULL CHECK (stage IN ('decomposition','synthesis')),
  attempt_no INTEGER NOT NULL CHECK (attempt_no>0),
  owner_run_id INTEGER NOT NULL,
  lease_epoch INTEGER NOT NULL CHECK (lease_epoch>0),
  lifecycle_revision INTEGER NOT NULL,
  cancel_epoch INTEGER NOT NULL CHECK (cancel_epoch>=0),
  attempt_state TEXT NOT NULL CHECK (attempt_state IN ('running','completed','blocked','failed','timed_out','cancelled')),
  started_at INTEGER NOT NULL,
  ended_at INTEGER,
  outcome_code TEXT,
  attempt_digest TEXT NOT NULL CHECK (length(attempt_digest)=64),
  PRIMARY KEY (board_instance_id,tenant_scope,orch_id,stage,attempt_no),
  UNIQUE (board_instance_id,tenant_scope,orch_id,stage,attempt_no,owner_run_id,
          lease_epoch,lifecycle_revision,cancel_epoch),
  UNIQUE (board_instance_id,tenant_scope,owner_run_id,stage),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id,stage)
    REFERENCES orch_stage_leases(board_instance_id,tenant_scope,orch_id,stage),
  CHECK ((attempt_state='running' AND ended_at IS NULL) OR
         (attempt_state!='running' AND ended_at IS NOT NULL AND ended_at>=started_at))
);

CREATE TABLE orch_results (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  orch_id TEXT NOT NULL,
  plan_version INTEGER NOT NULL,
  result_id TEXT NOT NULL,
  request_digest TEXT NOT NULL,
  plan_digest TEXT NOT NULL,
  producer_stage TEXT NOT NULL CHECK (producer_stage='synthesis'),
  producer_attempt_no INTEGER NOT NULL CHECK (producer_attempt_no>0),
  producer_run_id INTEGER NOT NULL,
  lifecycle_revision INTEGER NOT NULL,
  cancel_epoch INTEGER NOT NULL CHECK (cancel_epoch>=0),
  synthesis_epoch INTEGER NOT NULL CHECK (synthesis_epoch>0),
  accepted_lane_set_digest TEXT NOT NULL CHECK (length(accepted_lane_set_digest)=64),
  result_json TEXT NOT NULL,
  result_digest TEXT NOT NULL CHECK (length(result_digest)=64),
  accepted_at INTEGER NOT NULL,
  PRIMARY KEY (board_instance_id,tenant_scope,orch_id,result_id),
  UNIQUE (board_instance_id,tenant_scope,orch_id,plan_version),
  UNIQUE (board_instance_id,tenant_scope,orch_id,plan_version,result_id),
  UNIQUE (board_instance_id,tenant_scope,orch_id,plan_version,result_id,result_digest),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id,plan_version,request_digest,plan_digest)
    REFERENCES orch_plans(board_instance_id,tenant_scope,orch_id,plan_version,request_digest,plan_digest),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id,producer_stage,producer_attempt_no,
               producer_run_id,synthesis_epoch,lifecycle_revision,cancel_epoch)
    REFERENCES orch_stage_attempts(board_instance_id,tenant_scope,orch_id,stage,attempt_no,
                                   owner_run_id,lease_epoch,lifecycle_revision,cancel_epoch)
);

CREATE TABLE orch_delivery_manifests (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  orch_id TEXT NOT NULL,
  plan_version INTEGER NOT NULL,
  result_id TEXT NOT NULL,
  manifest_digest TEXT NOT NULL CHECK (length(manifest_digest)=64),
  expected_required_count INTEGER NOT NULL CHECK (expected_required_count>=0),
  expected_optional_count INTEGER NOT NULL CHECK (expected_optional_count>=0),
  lifecycle_revision INTEGER NOT NULL,
  cancel_epoch INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  PRIMARY KEY (board_instance_id,tenant_scope,orch_id,result_id),
  UNIQUE (board_instance_id,tenant_scope,orch_id,result_id,manifest_digest),
  UNIQUE (board_instance_id,tenant_scope,manifest_digest),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id,plan_version,result_id)
    REFERENCES orch_results(board_instance_id,tenant_scope,orch_id,plan_version,result_id)
);

CREATE TABLE orch_delivery_obligations (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  orch_id TEXT NOT NULL,
  plan_version INTEGER NOT NULL,
  result_id TEXT NOT NULL,
  obligation_id TEXT NOT NULL,
  manifest_digest TEXT NOT NULL CHECK (length(manifest_digest)=64),
  manifest_entry_key TEXT NOT NULL CHECK (length(manifest_entry_key)=64),
  delivery_generation INTEGER NOT NULL CHECK (delivery_generation>0),
  supersedes_obligation_id TEXT,
  delivery_key TEXT NOT NULL CHECK (length(delivery_key)=64),
  result_digest TEXT NOT NULL,
  origin_id TEXT NOT NULL,
  route_revision INTEGER NOT NULL CHECK (route_revision>0),
  route_digest TEXT NOT NULL CHECK (length(route_digest)=64),
  origin_kind TEXT NOT NULL CHECK (origin_kind IN ('messaging','api','local_session','board_only')),
  platform TEXT NOT NULL,
  adapter_instance_id TEXT NOT NULL,
  account_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  thread_id TEXT NOT NULL,
  reply_to_id TEXT NOT NULL,
  required INTEGER NOT NULL CHECK (required IN (0,1)),
  required_ack_family TEXT NOT NULL CHECK (required_ack_family IN (
    'provider','adapter','synchronous','local_session','none'
  )),
  required_ack_strength TEXT NOT NULL CHECK (required_ack_strength IN (
    'provider_message_id','adapter_acceptance','synchronous_response',
    'local_session_acceptance','none'
  )),
  state TEXT NOT NULL CHECK (state IN ('pending','claimed','accepted','acked','unknown','dead_letter','cancelled')),
  lifecycle_revision INTEGER NOT NULL,
  cancel_epoch INTEGER NOT NULL,
  claim_owner TEXT,
  claim_token_hash TEXT,
  claim_epoch INTEGER NOT NULL DEFAULT 0 CHECK (claim_epoch>=0),
  claim_expires_at INTEGER,
  attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts>=0),
  max_attempts INTEGER NOT NULL CHECK (max_attempts BETWEEN 1 AND 100),
  available_at INTEGER NOT NULL,
  acceptance_attempt_id TEXT,
  adapter_accepted_at INTEGER,
  acked_at INTEGER,
  provider_id TEXT,
  provider_message_id TEXT,
  duplicate_possible INTEGER NOT NULL DEFAULT 0 CHECK (duplicate_possible IN (0,1)),
  unknown_attempt_id TEXT,
  ambiguity_deadline INTEGER,
  resend_authorized_at INTEGER,
  resend_authorized_by TEXT,
  resend_authorized_revision INTEGER,
  last_error_code TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (board_instance_id,tenant_scope,obligation_id),
  UNIQUE (board_instance_id,tenant_scope,delivery_key),
  UNIQUE (board_instance_id,tenant_scope,manifest_entry_key,delivery_generation),
  UNIQUE (board_instance_id,tenant_scope,supersedes_obligation_id),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id,result_id,manifest_digest)
    REFERENCES orch_delivery_manifests(board_instance_id,tenant_scope,orch_id,result_id,manifest_digest),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id,plan_version,result_id,result_digest)
    REFERENCES orch_results(board_instance_id,tenant_scope,orch_id,plan_version,result_id,result_digest),
  FOREIGN KEY (board_instance_id,tenant_scope,origin_id,route_revision,route_digest)
    REFERENCES orch_origins(board_instance_id,tenant_scope,origin_id,route_revision,route_digest),
  FOREIGN KEY (board_instance_id,tenant_scope,supersedes_obligation_id)
    REFERENCES orch_delivery_obligations(board_instance_id,tenant_scope,obligation_id),
  CHECK ((origin_kind='board_only' AND required=0 AND required_ack_strength='none') OR
         (origin_kind!='board_only' AND required_ack_strength!='none')),
  CHECK ((state='acked' AND acked_at IS NOT NULL) OR state!='acked'),
  CHECK ((state IN ('claimed','accepted') AND claim_epoch>0) OR state NOT IN ('claimed','accepted')),
  CHECK ((state='unknown' AND unknown_attempt_id IS NOT NULL AND ambiguity_deadline IS NOT NULL)
      OR state!='unknown'),
  CHECK ((required_ack_family='provider' AND required_ack_strength='provider_message_id')
      OR (required_ack_family='adapter' AND required_ack_strength='adapter_acceptance')
      OR (required_ack_family='synchronous' AND required_ack_strength='synchronous_response')
      OR (required_ack_family='local_session' AND required_ack_strength='local_session_acceptance')
      OR (required_ack_family='none' AND required_ack_strength='none'))
);

CREATE TABLE orch_delivery_manifest_entries (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  orch_id TEXT NOT NULL,
  result_id TEXT NOT NULL,
  manifest_digest TEXT NOT NULL,
  manifest_entry_key TEXT NOT NULL,
  obligation_id TEXT NOT NULL,
  required INTEGER NOT NULL CHECK (required IN (0,1)),
  required_ack_family TEXT NOT NULL,
  required_ack_strength TEXT NOT NULL,
  route_digest TEXT NOT NULL,
  ordinal INTEGER NOT NULL CHECK (ordinal>0),
  PRIMARY KEY (board_instance_id,tenant_scope,manifest_entry_key),
  UNIQUE (board_instance_id,tenant_scope,orch_id,result_id,ordinal),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id,result_id,manifest_digest)
    REFERENCES orch_delivery_manifests(board_instance_id,tenant_scope,orch_id,result_id,manifest_digest),
  FOREIGN KEY (board_instance_id,tenant_scope,obligation_id)
    REFERENCES orch_delivery_obligations(board_instance_id,tenant_scope,obligation_id)
);

CREATE TABLE orch_delivery_attempts (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  obligation_id TEXT NOT NULL,
  attempt_id TEXT NOT NULL,
  claim_epoch INTEGER NOT NULL CHECK (claim_epoch>0),
  claim_owner TEXT NOT NULL,
  claim_token_hash TEXT NOT NULL CHECK (length(claim_token_hash)=64),
  lifecycle_revision INTEGER NOT NULL,
  cancel_epoch INTEGER NOT NULL,
  send_nonce TEXT NOT NULL,
  payload_digest TEXT NOT NULL CHECK (length(payload_digest)=64),
  result_digest TEXT NOT NULL CHECK (length(result_digest)=64),
  route_digest TEXT NOT NULL CHECK (length(route_digest)=64),
  attempt_state TEXT NOT NULL CHECK (attempt_state IN ('started','adapter_accepted','rejected','unknown')),
  started_at INTEGER NOT NULL,
  finished_at INTEGER,
  adapter_evidence_json TEXT,
  adapter_evidence_digest TEXT,
  error_code TEXT,
  PRIMARY KEY (board_instance_id,tenant_scope,attempt_id),
  UNIQUE (board_instance_id,tenant_scope,obligation_id,claim_epoch),
  FOREIGN KEY (board_instance_id,tenant_scope,obligation_id)
    REFERENCES orch_delivery_obligations(board_instance_id,tenant_scope,obligation_id),
  CHECK ((attempt_state='started' AND finished_at IS NULL) OR
         (attempt_state!='started' AND finished_at IS NOT NULL))
);

CREATE TABLE orch_delivery_attempt_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  attempt_id TEXT NOT NULL,
  transition_seq INTEGER NOT NULL CHECK (transition_seq>0),
  from_state TEXT,
  to_state TEXT NOT NULL CHECK (to_state IN ('started','adapter_accepted','rejected','unknown')),
  event_kind TEXT NOT NULL CHECK (event_kind IN ('started','adapter_accepted','rejected','unknown')),
  event_json TEXT NOT NULL,
  event_digest TEXT NOT NULL CHECK (length(event_digest)=64),
  created_at INTEGER NOT NULL,
  UNIQUE (board_instance_id,tenant_scope,attempt_id,transition_seq),
  UNIQUE (board_instance_id,tenant_scope,attempt_id,event_kind,event_digest),
  FOREIGN KEY (board_instance_id,tenant_scope,attempt_id)
    REFERENCES orch_delivery_attempts(board_instance_id,tenant_scope,attempt_id)
);

CREATE TABLE orch_delivery_receipts (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  obligation_id TEXT NOT NULL,
  attempt_id TEXT NOT NULL,
  receipt_id TEXT NOT NULL,
  observed_ack_family TEXT NOT NULL CHECK (observed_ack_family IN (
    'provider','adapter','synchronous','local_session'
  )),
  observed_ack_strength TEXT NOT NULL CHECK (observed_ack_strength IN (
    'provider_message_id','adapter_acceptance','synchronous_response','local_session_acceptance'
  )),
  receipt_json TEXT NOT NULL,
  receipt_digest TEXT NOT NULL CHECK (length(receipt_digest)=64),
  send_nonce TEXT NOT NULL,
  payload_digest TEXT NOT NULL CHECK (length(payload_digest)=64),
  result_digest TEXT NOT NULL CHECK (length(result_digest)=64),
  route_digest TEXT NOT NULL CHECK (length(route_digest)=64),
  verified INTEGER NOT NULL CHECK (verified=1),
  provider_id TEXT,
  provider_message_id TEXT,
  created_at INTEGER NOT NULL,
  PRIMARY KEY (board_instance_id,tenant_scope,receipt_id),
  UNIQUE (board_instance_id,tenant_scope,obligation_id),
  FOREIGN KEY (board_instance_id,tenant_scope,attempt_id)
    REFERENCES orch_delivery_attempts(board_instance_id,tenant_scope,attempt_id),
  FOREIGN KEY (board_instance_id,tenant_scope,obligation_id)
    REFERENCES orch_delivery_obligations(board_instance_id,tenant_scope,obligation_id),
  CHECK ((observed_ack_family='provider' AND observed_ack_strength='provider_message_id'
          AND provider_id IS NOT NULL AND provider_message_id IS NOT NULL)
      OR (observed_ack_family='adapter' AND observed_ack_strength='adapter_acceptance')
      OR (observed_ack_family='synchronous' AND observed_ack_strength='synchronous_response')
      OR (observed_ack_family='local_session' AND observed_ack_strength='local_session_acceptance'))
);

CREATE TABLE orch_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  orch_id TEXT NOT NULL,
  lifecycle_revision INTEGER NOT NULL,
  cancel_epoch INTEGER NOT NULL,
  commit_seq INTEGER NOT NULL CHECK (commit_seq>0),
  event_kind TEXT NOT NULL CHECK (event_kind IN (
    'request_transition','node_transition','plan_materialized','result_accepted',
    'delivery_transition','cancellation_transition','dependency_released'
  )),
  target_key TEXT NOT NULL,
  event_key TEXT NOT NULL CHECK (length(event_key)=64),
  payload_json TEXT NOT NULL,
  payload_digest TEXT NOT NULL CHECK (length(payload_digest)=64),
  created_at INTEGER NOT NULL,
  UNIQUE (board_instance_id,tenant_scope,event_key),
  UNIQUE (board_instance_id,tenant_scope,event_id),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id)
    REFERENCES orch_requests(board_instance_id,tenant_scope,orch_id)
);

CREATE TABLE orch_reconcile_queue (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  event_id INTEGER NOT NULL,
  consumer_kind TEXT NOT NULL CHECK (consumer_kind IN ('lifecycle','delivery','dependency','cancellation')),
  state TEXT NOT NULL CHECK (state IN ('pending','claimed','done','dead_letter')),
  claim_owner TEXT,
  claim_token_hash TEXT,
  claim_epoch INTEGER NOT NULL DEFAULT 0 CHECK (claim_epoch>=0),
  claim_expires_at INTEGER,
  available_at INTEGER NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts>=0),
  max_attempts INTEGER NOT NULL CHECK (max_attempts BETWEEN 1 AND 100),
  last_error_code TEXT,
  done_effect_digest TEXT,
  PRIMARY KEY (board_instance_id,tenant_scope,event_id,consumer_kind),
  FOREIGN KEY (board_instance_id,tenant_scope,event_id)
    REFERENCES orch_events(board_instance_id,tenant_scope,event_id),
  CHECK ((state='claimed' AND claim_owner IS NOT NULL AND claim_token_hash IS NOT NULL AND claim_epoch>0)
      OR (state!='claimed')),
  CHECK ((state='done' AND length(done_effect_digest)=64) OR state!='done')
);

CREATE TABLE orch_effect_ledger (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  event_id INTEGER NOT NULL,
  consumer_kind TEXT NOT NULL,
  target_key TEXT NOT NULL,
  target_revision INTEGER NOT NULL,
  effect_digest TEXT NOT NULL CHECK (length(effect_digest)=64),
  applied_at INTEGER NOT NULL,
  PRIMARY KEY (board_instance_id,tenant_scope,event_id,consumer_kind),
  UNIQUE (board_instance_id,tenant_scope,consumer_kind,target_key,target_revision),
  FOREIGN KEY (board_instance_id,tenant_scope,event_id,consumer_kind)
    REFERENCES orch_reconcile_queue(board_instance_id,tenant_scope,event_id,consumer_kind)
);

CREATE TABLE orch_delivery_resend_authorizations (
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  authorization_id TEXT NOT NULL,
  obligation_id TEXT NOT NULL,
  unknown_attempt_id TEXT NOT NULL,
  lifecycle_revision INTEGER NOT NULL,
  cancel_epoch INTEGER NOT NULL,
  capability_digest TEXT NOT NULL CHECK (length(capability_digest)=64),
  authorized_by TEXT NOT NULL,
  expires_at INTEGER NOT NULL,
  consumed_at INTEGER,
  created_at INTEGER NOT NULL,
  PRIMARY KEY (board_instance_id,tenant_scope,authorization_id),
  UNIQUE (board_instance_id,tenant_scope,obligation_id,unknown_attempt_id),
  FOREIGN KEY (board_instance_id,tenant_scope,obligation_id)
    REFERENCES orch_delivery_obligations(board_instance_id,tenant_scope,obligation_id),
  FOREIGN KEY (board_instance_id,tenant_scope,unknown_attempt_id)
    REFERENCES orch_delivery_attempts(board_instance_id,tenant_scope,attempt_id)
);

CREATE TABLE orch_mutation_log (
  mutation_id INTEGER PRIMARY KEY AUTOINCREMENT,
  board_instance_id TEXT NOT NULL,
  tenant_scope TEXT NOT NULL,
  orch_id TEXT NOT NULL,
  stage TEXT NOT NULL,
  owner_run_id INTEGER NOT NULL,
  owner_profile TEXT NOT NULL,
  lease_epoch INTEGER NOT NULL,
  lifecycle_revision INTEGER NOT NULL,
  cancel_epoch INTEGER NOT NULL,
  capability_digest TEXT NOT NULL CHECK (length(capability_digest)=64),
  commit_seq INTEGER NOT NULL CHECK (commit_seq>0),
  mutation_kind TEXT NOT NULL,
  target_key TEXT NOT NULL,
  mutation_digest TEXT NOT NULL CHECK (length(mutation_digest)=64),
  created_at INTEGER NOT NULL,
  UNIQUE (board_instance_id,tenant_scope,orch_id,mutation_kind,target_key,mutation_digest),
  FOREIGN KEY (board_instance_id,tenant_scope,orch_id)
    REFERENCES orch_requests(board_instance_id,tenant_scope,orch_id)
);

CREATE TRIGGER immutable_orch_origins_u BEFORE UPDATE ON orch_origins BEGIN SELECT RAISE(ABORT,'immutable_orch_origin'); END;
CREATE TRIGGER immutable_orch_origins_d BEFORE DELETE ON orch_origins BEGIN SELECT RAISE(ABORT,'immutable_orch_origin'); END;
CREATE TRIGGER immutable_orch_requirements_u BEFORE UPDATE ON orch_request_requirements BEGIN SELECT RAISE(ABORT,'immutable_orch_requirement'); END;
CREATE TRIGGER immutable_orch_requirements_d BEFORE DELETE ON orch_request_requirements BEGIN SELECT RAISE(ABORT,'immutable_orch_requirement'); END;
CREATE TRIGGER immutable_orch_plans_u BEFORE UPDATE ON orch_plans BEGIN SELECT RAISE(ABORT,'immutable_orch_plan'); END;
CREATE TRIGGER immutable_orch_plans_d BEFORE DELETE ON orch_plans BEGIN SELECT RAISE(ABORT,'immutable_orch_plan'); END;
CREATE TRIGGER immutable_orch_plan_nodes_u BEFORE UPDATE ON orch_plan_nodes BEGIN SELECT RAISE(ABORT,'immutable_orch_plan_node'); END;
CREATE TRIGGER immutable_orch_plan_nodes_d BEFORE DELETE ON orch_plan_nodes BEGIN SELECT RAISE(ABORT,'immutable_orch_plan_node'); END;
CREATE TRIGGER immutable_orch_plan_edges_u BEFORE UPDATE ON orch_plan_edges BEGIN SELECT RAISE(ABORT,'immutable_orch_plan_edge'); END;
CREATE TRIGGER immutable_orch_plan_edges_d BEFORE DELETE ON orch_plan_edges BEGIN SELECT RAISE(ABORT,'immutable_orch_plan_edge'); END;
CREATE TRIGGER immutable_orch_plan_coverage_u BEFORE UPDATE ON orch_plan_coverage BEGIN SELECT RAISE(ABORT,'immutable_orch_plan_coverage'); END;
CREATE TRIGGER immutable_orch_plan_coverage_d BEFORE DELETE ON orch_plan_coverage BEGIN SELECT RAISE(ABORT,'immutable_orch_plan_coverage'); END;
CREATE TRIGGER immutable_orch_materialization_u BEFORE UPDATE ON orch_plan_materializations BEGIN SELECT RAISE(ABORT,'immutable_orch_materialization'); END;
CREATE TRIGGER immutable_orch_materialization_d BEFORE DELETE ON orch_plan_materializations BEGIN SELECT RAISE(ABORT,'immutable_orch_materialization'); END;
CREATE TRIGGER immutable_orch_acceptance_u BEFORE UPDATE ON orch_node_acceptances BEGIN SELECT RAISE(ABORT,'immutable_orch_acceptance'); END;
CREATE TRIGGER immutable_orch_acceptance_d BEFORE DELETE ON orch_node_acceptances BEGIN SELECT RAISE(ABORT,'immutable_orch_acceptance'); END;
CREATE TRIGGER immutable_orch_result_u BEFORE UPDATE ON orch_results BEGIN SELECT RAISE(ABORT,'immutable_orch_result'); END;
CREATE TRIGGER immutable_orch_result_d BEFORE DELETE ON orch_results BEGIN SELECT RAISE(ABORT,'immutable_orch_result'); END;
CREATE TRIGGER immutable_orch_delivery_attempt_event_u BEFORE UPDATE ON orch_delivery_attempt_events BEGIN SELECT RAISE(ABORT,'immutable_orch_delivery_attempt_event'); END;
CREATE TRIGGER immutable_orch_delivery_attempt_event_d BEFORE DELETE ON orch_delivery_attempt_events BEGIN SELECT RAISE(ABORT,'immutable_orch_delivery_attempt_event'); END;
CREATE TRIGGER immutable_orch_delivery_receipt_u BEFORE UPDATE ON orch_delivery_receipts BEGIN SELECT RAISE(ABORT,'immutable_orch_delivery_receipt'); END;
CREATE TRIGGER immutable_orch_delivery_receipt_d BEFORE DELETE ON orch_delivery_receipts BEGIN SELECT RAISE(ABORT,'immutable_orch_delivery_receipt'); END;
CREATE TRIGGER immutable_orch_event_u BEFORE UPDATE ON orch_events BEGIN SELECT RAISE(ABORT,'immutable_orch_event'); END;
CREATE TRIGGER immutable_orch_event_d BEFORE DELETE ON orch_events BEGIN SELECT RAISE(ABORT,'immutable_orch_event'); END;
CREATE TRIGGER immutable_orch_mutation_log_u BEFORE UPDATE ON orch_mutation_log BEGIN SELECT RAISE(ABORT,'immutable_orch_mutation_log'); END;
CREATE TRIGGER immutable_orch_mutation_log_d BEFORE DELETE ON orch_mutation_log BEGIN SELECT RAISE(ABORT,'immutable_orch_mutation_log'); END;

CREATE TRIGGER orch_event_insert_guard BEFORE INSERT ON orch_events
BEGIN
  SELECT CASE WHEN orch_capability_ok(
    'event_insert',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,
    NEW.lifecycle_revision,NEW.cancel_epoch,NEW.event_key
  )!=1 OR NOT EXISTS (
    SELECT 1 FROM orch_requests r JOIN kanban_commit_clock c ON c.singleton=1
     WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope
       AND r.orch_id=NEW.orch_id AND r.lifecycle_revision=NEW.lifecycle_revision
       AND r.cancel_epoch=NEW.cancel_epoch AND c.commit_seq=NEW.commit_seq
  ) THEN RAISE(ABORT,'stale_or_forged_orch_event') END;
END;
CREATE TRIGGER orch_queue_insert_guard BEFORE INSERT ON orch_reconcile_queue
BEGIN
  SELECT CASE WHEN NEW.state!='pending' OR orch_capability_ok(
    'queue_fanout',NEW.board_instance_id,NEW.tenant_scope,CAST(NEW.event_id AS TEXT),
    0,0,NEW.consumer_kind
  )!=1 THEN RAISE(ABORT,'forged_reconcile_queue_row') END;
END;
CREATE TRIGGER orch_effect_insert_guard BEFORE INSERT ON orch_effect_ledger
BEGIN
  SELECT CASE WHEN orch_capability_ok(
    'effect_apply',NEW.board_instance_id,NEW.tenant_scope,CAST(NEW.event_id AS TEXT),
    NEW.target_revision,0,NEW.effect_digest
  )!=1 THEN RAISE(ABORT,'forged_effect_ledger_row') END;
END;

CREATE TRIGGER orch_materialization_authority_guard BEFORE INSERT ON orch_plan_materializations
BEGIN
  SELECT CASE WHEN orch_capability_ok(
    'materialize',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,
    NEW.request_lifecycle_revision,NEW.cancel_epoch,NEW.observed_graph_digest
  )!=1 THEN RAISE(ABORT,'missing_materialization_capability') END;
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM orch_requests r
    JOIN orch_plans p
      ON p.board_instance_id=r.board_instance_id AND p.tenant_scope=r.tenant_scope
     AND p.orch_id=r.orch_id AND p.plan_version=NEW.plan_version
    JOIN orch_stage_leases l
      ON l.board_instance_id=r.board_instance_id AND l.tenant_scope=r.tenant_scope
     AND l.orch_id=r.orch_id AND l.stage='decomposition'
    JOIN kanban_commit_clock c ON c.singleton=1
    WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope
      AND r.orch_id=NEW.orch_id AND r.lifecycle_state='decomposing'
      AND r.plan_version+1=NEW.plan_version
      AND r.lifecycle_revision=NEW.request_lifecycle_revision
      AND r.cancel_epoch=NEW.cancel_epoch
      AND p.plan_digest=NEW.plan_digest
      AND l.owner_run_id=NEW.materialized_by_run_id AND l.epoch=NEW.lease_epoch
      AND l.lease_state='active' AND l.expires_at>unixepoch('now')
      AND c.commit_seq=NEW.commit_seq
      AND NEW.observed_node_count=(SELECT count(*) FROM orch_nodes n
        WHERE n.board_instance_id=NEW.board_instance_id AND n.tenant_scope=NEW.tenant_scope
          AND n.orch_id=NEW.orch_id AND n.plan_version=NEW.plan_version)
      AND NEW.observed_edge_count=(SELECT count(*) FROM task_links x
        WHERE x.orch_board_instance_id=NEW.board_instance_id
          AND x.orch_tenant_scope=NEW.tenant_scope AND x.orch_id=NEW.orch_id
          AND x.orch_plan_version=NEW.plan_version)
  ) THEN RAISE(ABORT,'invalid_materialization_provenance') END;
END;

CREATE TRIGGER tasks_orch_binding_insert BEFORE INSERT ON tasks
WHEN NEW.orch_board_instance_id IS NOT NULL OR NEW.orch_tenant_scope IS NOT NULL
  OR NEW.orch_id IS NOT NULL OR NEW.orch_plan_version IS NOT NULL
  OR NEW.orch_node_key IS NOT NULL OR NEW.orch_binding_revision IS NOT NULL
  OR NEW.orch_cancel_epoch IS NOT NULL
BEGIN
  SELECT CASE WHEN NEW.orch_board_instance_id IS NULL OR NEW.orch_tenant_scope IS NULL
    OR NEW.orch_id IS NULL OR NEW.orch_plan_version IS NULL OR NEW.orch_node_key IS NULL
    OR NEW.orch_binding_revision IS NULL OR NEW.orch_cancel_epoch IS NULL
    THEN RAISE(ABORT,'partial_orch_task_binding') END;
  SELECT CASE WHEN orch_capability_ok(
      'task_bind',NEW.orch_board_instance_id,NEW.orch_tenant_scope,NEW.orch_id,
      NEW.orch_binding_revision,NEW.orch_cancel_epoch,NEW.id
    )!=1 THEN RAISE(ABORT,'missing_task_bind_capability') END;
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM orch_requests r
     WHERE r.board_instance_id=NEW.orch_board_instance_id
       AND r.tenant_scope=NEW.orch_tenant_scope AND r.orch_id=NEW.orch_id
       AND r.parent_task_id=NEW.id AND NEW.orch_node_key='__parent__'
       AND NEW.orch_plan_version=r.plan_version
       AND r.lifecycle_revision=NEW.orch_binding_revision
       AND r.cancel_epoch=NEW.orch_cancel_epoch
    UNION ALL
    SELECT 1 FROM orch_nodes n JOIN orch_requests r
      ON r.board_instance_id=n.board_instance_id AND r.tenant_scope=n.tenant_scope
     AND r.orch_id=n.orch_id
     WHERE n.board_instance_id=NEW.orch_board_instance_id
       AND n.tenant_scope=NEW.orch_tenant_scope AND n.orch_id=NEW.orch_id
       AND n.plan_version=NEW.orch_plan_version AND n.node_key=NEW.orch_node_key
       AND n.task_id=NEW.id AND r.lifecycle_revision=NEW.orch_binding_revision
       AND r.cancel_epoch=NEW.orch_cancel_epoch
  ) THEN RAISE(ABORT,'invalid_orch_task_scope') END;
END;

CREATE TRIGGER tasks_orch_binding_update BEFORE UPDATE OF
  orch_board_instance_id,orch_tenant_scope,orch_id,orch_plan_version,
  orch_node_key,orch_binding_revision,orch_cancel_epoch ON tasks
WHEN OLD.orch_id IS NOT NULL OR NEW.orch_id IS NOT NULL
BEGIN
  SELECT CASE WHEN NOT (
    (NEW.orch_board_instance_id IS OLD.orch_board_instance_id
      AND NEW.orch_tenant_scope IS OLD.orch_tenant_scope
      AND NEW.orch_id IS OLD.orch_id
      AND NEW.orch_plan_version IS OLD.orch_plan_version
      AND NEW.orch_node_key IS OLD.orch_node_key
      AND NEW.orch_binding_revision IS OLD.orch_binding_revision
      AND NEW.orch_cancel_epoch IS OLD.orch_cancel_epoch)
    OR
    (OLD.orch_board_instance_id IS NULL AND OLD.orch_tenant_scope IS NULL
      AND OLD.orch_id IS NULL AND OLD.orch_plan_version IS NULL
      AND OLD.orch_node_key IS NULL AND OLD.orch_binding_revision IS NULL
      AND OLD.orch_cancel_epoch IS NULL
      AND NEW.orch_node_key='__parent__'
      AND EXISTS (SELECT 1 FROM orch_requests r
        WHERE r.board_instance_id=NEW.orch_board_instance_id
          AND r.tenant_scope=NEW.orch_tenant_scope AND r.orch_id=NEW.orch_id
          AND r.parent_task_id=NEW.id AND r.plan_version=NEW.orch_plan_version
          AND r.lifecycle_revision=NEW.orch_binding_revision
          AND r.cancel_epoch=NEW.orch_cancel_epoch)
      AND orch_capability_ok('task_bind',NEW.orch_board_instance_id,NEW.orch_tenant_scope,NEW.orch_id,
        NEW.orch_binding_revision,NEW.orch_cancel_epoch,NEW.id)=1)
    OR
    (NEW.orch_board_instance_id IS NULL AND NEW.orch_tenant_scope IS NULL
      AND NEW.orch_id IS NULL AND NEW.orch_plan_version IS NULL
      AND NEW.orch_node_key IS NULL AND NEW.orch_binding_revision IS NULL
      AND NEW.orch_cancel_epoch IS NULL
      AND EXISTS (SELECT 1 FROM orch_requests r
        WHERE r.board_instance_id=OLD.orch_board_instance_id
          AND r.tenant_scope=OLD.orch_tenant_scope AND r.orch_id=OLD.orch_id
          AND r.lifecycle_state IN ('cancelling','cancelled'))
      AND orch_capability_ok(
        'task_retire',OLD.orch_board_instance_id,OLD.orch_tenant_scope,OLD.orch_id,
        COALESCE((SELECT lifecycle_revision FROM orch_requests r
          WHERE r.board_instance_id=OLD.orch_board_instance_id AND r.tenant_scope=OLD.orch_tenant_scope
            AND r.orch_id=OLD.orch_id),-1),
        COALESCE((SELECT cancel_epoch FROM orch_requests r
          WHERE r.board_instance_id=OLD.orch_board_instance_id AND r.tenant_scope=OLD.orch_tenant_scope
            AND r.orch_id=OLD.orch_id),-1),OLD.id)=1)
  ) THEN RAISE(ABORT,'immutable_orch_task_binding') END;
END;

CREATE TRIGGER orch_acceptance_run_guard BEFORE INSERT ON orch_node_acceptances
BEGIN
  SELECT CASE WHEN orch_capability_ok(
    'accept_lane',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,
    NEW.plan_epoch_revision,NEW.cancel_epoch,CAST(NEW.accepted_run_id AS TEXT)
  )!=1 THEN RAISE(ABORT,'missing_acceptance_capability') END;
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1
      FROM orch_nodes n
      JOIN orch_plan_nodes pn
        ON pn.board_instance_id=n.board_instance_id AND pn.tenant_scope=n.tenant_scope
       AND pn.orch_id=n.orch_id AND pn.plan_version=n.plan_version AND pn.node_key=n.node_key
      JOIN task_runs tr ON tr.id=NEW.accepted_run_id AND tr.task_id=n.task_id
      JOIN orch_requests r
        ON r.board_instance_id=n.board_instance_id AND r.tenant_scope=n.tenant_scope AND r.orch_id=n.orch_id
      JOIN orch_stage_leases l
        ON l.board_instance_id=r.board_instance_id AND l.tenant_scope=r.tenant_scope
       AND l.orch_id=r.orch_id AND l.stage='reconciliation'
     WHERE n.board_instance_id=NEW.board_instance_id
       AND n.tenant_scope=NEW.tenant_scope AND n.orch_id=NEW.orch_id
       AND n.plan_version=NEW.plan_version AND n.node_key=NEW.node_key AND n.task_id=NEW.task_id
       AND pn.role='lane' AND pn.required=1
       AND r.plan_version=NEW.plan_version AND r.lifecycle_state='waiting_lanes'
       AND r.plan_epoch_revision=NEW.plan_epoch_revision AND r.cancel_epoch=NEW.cancel_epoch
       AND l.owner_run_id=NEW.accepted_by_run_id AND l.epoch=NEW.acceptance_lease_epoch
       AND l.lease_state='active' AND l.expires_at>unixepoch('now')
       AND tr.status='done' AND tr.outcome='completed'
       AND tr.started_at IS NOT NULL AND tr.ended_at IS NOT NULL
       AND tr.ended_at>=tr.started_at AND tr.outcome_digest=NEW.outcome_digest
       AND tr.cancellation_epoch=NEW.cancel_epoch
  ) THEN RAISE(ABORT,'invalid_accepted_run_binding') END;
END;

CREATE TRIGGER task_links_orch_guard BEFORE INSERT ON task_links
WHEN EXISTS (SELECT 1 FROM tasks t WHERE t.id IN (NEW.parent_id,NEW.child_id) AND t.orch_id IS NOT NULL)
BEGIN
  SELECT CASE WHEN NEW.orch_board_instance_id IS NULL OR NEW.orch_tenant_scope IS NULL
    OR NEW.orch_id IS NULL OR NEW.orch_plan_version IS NULL OR NEW.orch_edge_key IS NULL
    OR NEW.orch_binding_revision IS NULL OR NEW.orch_cancel_epoch IS NULL
    THEN RAISE(ABORT,'unregistered_orch_link') END;
  SELECT CASE WHEN orch_capability_ok(
    'link_bind',NEW.orch_board_instance_id,NEW.orch_tenant_scope,NEW.orch_id,
    NEW.orch_binding_revision,NEW.orch_cancel_epoch,NEW.orch_edge_key
  )!=1 THEN RAISE(ABORT,'missing_link_capability') END;
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1
      FROM orch_plan_edges e
      JOIN orch_nodes p
        ON p.board_instance_id=e.board_instance_id AND p.tenant_scope=e.tenant_scope
       AND p.orch_id=e.orch_id AND p.plan_version=e.plan_version
       AND p.node_key=e.parent_node_key AND p.task_id=NEW.parent_id
      JOIN orch_nodes c
        ON c.board_instance_id=e.board_instance_id AND c.tenant_scope=e.tenant_scope
       AND c.orch_id=e.orch_id AND c.plan_version=e.plan_version
       AND c.node_key=e.child_node_key AND c.task_id=NEW.child_id
      JOIN orch_requests r
        ON r.board_instance_id=e.board_instance_id AND r.tenant_scope=e.tenant_scope AND r.orch_id=e.orch_id
     WHERE e.board_instance_id=NEW.orch_board_instance_id
       AND e.tenant_scope=NEW.orch_tenant_scope AND e.orch_id=NEW.orch_id
       AND e.plan_version=NEW.orch_plan_version AND e.edge_key=NEW.orch_edge_key
       AND NEW.kind=e.edge_kind AND r.lifecycle_revision=NEW.orch_binding_revision
       AND r.cancel_epoch=NEW.orch_cancel_epoch
    UNION ALL
    SELECT 1 FROM orch_external_edges x JOIN orch_requests r
      ON r.board_instance_id=x.board_instance_id AND r.tenant_scope=x.tenant_scope AND r.orch_id=x.orch_id
     WHERE x.board_instance_id=NEW.orch_board_instance_id
       AND x.tenant_scope=NEW.orch_tenant_scope AND x.orch_id=NEW.orch_id
       AND x.edge_key=NEW.orch_edge_key AND x.parent_task_id=NEW.parent_id
       AND x.child_task_id=NEW.child_id AND NEW.kind=x.edge_kind
       AND r.lifecycle_revision=NEW.orch_binding_revision
       AND r.cancel_epoch=NEW.orch_cancel_epoch
  ) THEN RAISE(ABORT,'invalid_orch_link_binding') END;
END;

CREATE TRIGGER orch_request_identity_guard BEFORE UPDATE OF
  board_instance_id,tenant_scope,orch_id,lineage_id,generation,selector_key,selector_ledger_revision,request_key,
  request_schema_version,request_json,request_digest,origin_id,parent_task_id,
  synthesis_strategy,supersedes_orch_id,created_at
ON orch_requests BEGIN SELECT RAISE(ABORT,'immutable_orch_request_identity'); END;
CREATE TRIGGER orch_request_delete_guard BEFORE DELETE ON orch_requests
BEGIN SELECT RAISE(ABORT,'orch_request_delete_forbidden'); END;

CREATE TRIGGER orch_request_supersession_guard BEFORE INSERT ON orch_requests
BEGIN
  SELECT CASE WHEN NEW.lifecycle_state!='submitted' OR NEW.lifecycle_revision!=0
    OR NEW.cancel_epoch!=0 OR NEW.delivery_epoch_revision!=0 OR NEW.plan_epoch_revision!=0
    OR NEW.plan_version!=0 OR NEW.work_accepted_at IS NOT NULL OR NEW.delivery_closed_at IS NOT NULL
    OR NEW.terminal_reason_code IS NOT NULL OR NEW.blocked_from_state IS NOT NULL
    OR NEW.resume_state IS NOT NULL OR NEW.block_kind IS NOT NULL
    THEN RAISE(ABORT,'invalid_initial_orch_request_state') END;
  SELECT CASE WHEN orch_capability_ok(
    CASE WHEN NEW.generation=1 THEN 'request_submit' ELSE 'request_supersede' END,
    NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,
    NEW.selector_ledger_revision,NEW.generation,NEW.request_key
  )!=1 THEN RAISE(ABORT,'missing_request_capability') END;
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM orch_origins o JOIN orch_replay_selectors s
      ON s.board_instance_id=o.board_instance_id AND s.tenant_scope=o.tenant_scope
     AND s.selector_key=o.selector_key
     WHERE o.board_instance_id=NEW.board_instance_id AND o.tenant_scope=NEW.tenant_scope
       AND o.origin_id=NEW.origin_id AND o.selector_key=NEW.selector_key
       AND s.lineage_id=NEW.lineage_id AND s.ledger_revision=NEW.selector_ledger_revision
  ) THEN RAISE(ABORT,'request_selector_origin_mismatch') END;
  SELECT CASE WHEN NEW.generation=1 AND (
    NEW.supersedes_orch_id IS NOT NULL OR NOT EXISTS (
      SELECT 1 FROM orch_replay_selectors s
       WHERE s.board_instance_id=NEW.board_instance_id AND s.tenant_scope=NEW.tenant_scope
         AND s.selector_key=NEW.selector_key AND s.current_generation=0
         AND s.current_orch_id IS NULL
    )
  ) THEN RAISE(ABORT,'invalid_generation_one_request') END;
  SELECT CASE WHEN NEW.generation>1 AND NOT EXISTS (
    SELECT 1 FROM orch_requests prev JOIN orch_replay_selectors s
      ON s.board_instance_id=prev.board_instance_id AND s.tenant_scope=prev.tenant_scope
     AND s.selector_key=prev.selector_key
     WHERE prev.board_instance_id=NEW.board_instance_id
       AND prev.tenant_scope=NEW.tenant_scope AND prev.lineage_id=NEW.lineage_id
       AND prev.selector_key=NEW.selector_key AND prev.generation=NEW.generation-1
       AND prev.orch_id=NEW.supersedes_orch_id
       AND prev.lifecycle_state IN ('failed','cancelled')
       AND s.current_generation=prev.generation AND s.current_orch_id=prev.orch_id
       AND s.current_request_digest=prev.request_digest
       AND s.ledger_revision=NEW.selector_ledger_revision
  ) THEN RAISE(ABORT,'invalid_orch_supersession') END;
END;

CREATE TRIGGER orch_node_identity_guard BEFORE UPDATE OF
  board_instance_id,tenant_scope,orch_id,plan_version,node_key,task_id,
  current_route_digest,route_revision,created_by_run_id,created_at
ON orch_nodes BEGIN SELECT RAISE(ABORT,'immutable_orch_node_identity'); END;
CREATE TRIGGER orch_node_delete_guard BEFORE DELETE ON orch_nodes
BEGIN SELECT RAISE(ABORT,'orch_node_delete_forbidden'); END;

CREATE TRIGGER orch_stage_attempt_guard BEFORE INSERT ON orch_stage_attempts
BEGIN
  SELECT CASE WHEN orch_capability_ok(
    'stage_attempt_start',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,
    NEW.lifecycle_revision,NEW.cancel_epoch,NEW.attempt_digest
  )!=1 THEN RAISE(ABORT,'missing_stage_attempt_capability') END;
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM orch_stage_leases l JOIN orch_requests r
      ON r.board_instance_id=l.board_instance_id AND r.tenant_scope=l.tenant_scope AND r.orch_id=l.orch_id
     WHERE l.board_instance_id=NEW.board_instance_id AND l.tenant_scope=NEW.tenant_scope
       AND l.orch_id=NEW.orch_id AND l.stage=NEW.stage
       AND l.owner_run_id=NEW.owner_run_id AND l.epoch=NEW.lease_epoch
       AND l.lease_state='active' AND l.expires_at>unixepoch('now')
       AND r.lifecycle_revision=NEW.lifecycle_revision AND r.cancel_epoch=NEW.cancel_epoch
       AND NEW.attempt_state='running'
       AND NEW.attempt_no=(SELECT COALESCE(max(x.attempt_no),0)+1 FROM orch_stage_attempts x
         WHERE x.board_instance_id=NEW.board_instance_id AND x.tenant_scope=NEW.tenant_scope
           AND x.orch_id=NEW.orch_id AND x.stage=NEW.stage)
       AND NEW.attempt_no<=r.max_retries+1
       AND ((NEW.stage='decomposition' AND r.lifecycle_state='decomposing')
         OR (NEW.stage='synthesis' AND r.lifecycle_state='synthesizing'))
  ) THEN RAISE(ABORT,'invalid_orch_stage_attempt_authority') END;
END;

CREATE TRIGGER orch_stage_attempt_update_guard BEFORE UPDATE ON orch_stage_attempts
BEGIN
  SELECT CASE WHEN OLD.board_instance_id!=NEW.board_instance_id OR OLD.tenant_scope!=NEW.tenant_scope
    OR OLD.orch_id!=NEW.orch_id OR OLD.stage!=NEW.stage OR OLD.attempt_no!=NEW.attempt_no
    OR OLD.owner_run_id!=NEW.owner_run_id OR OLD.lease_epoch!=NEW.lease_epoch
    OR OLD.lifecycle_revision!=NEW.lifecycle_revision OR OLD.cancel_epoch!=NEW.cancel_epoch
    OR OLD.started_at!=NEW.started_at OR OLD.attempt_digest!=NEW.attempt_digest
    THEN RAISE(ABORT,'immutable_stage_attempt_identity') END;
  SELECT CASE WHEN OLD.attempt_state!='running' OR NEW.attempt_state='running'
    OR orch_capability_ok(
      'stage_attempt_finish',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,
      NEW.lifecycle_revision,NEW.cancel_epoch,NEW.attempt_digest
    )!=1 OR NOT EXISTS (
      SELECT 1 FROM orch_stage_leases l JOIN orch_requests r
        ON r.board_instance_id=l.board_instance_id AND r.tenant_scope=l.tenant_scope AND r.orch_id=l.orch_id
       WHERE l.board_instance_id=NEW.board_instance_id AND l.tenant_scope=NEW.tenant_scope
         AND l.orch_id=NEW.orch_id AND l.stage=NEW.stage
         AND l.owner_run_id=NEW.owner_run_id AND l.epoch=NEW.lease_epoch
         AND l.lease_state='active' AND l.expires_at>unixepoch('now')
         AND r.lifecycle_revision=NEW.lifecycle_revision AND r.cancel_epoch=NEW.cancel_epoch
    ) THEN RAISE(ABORT,'invalid_stage_attempt_transition') END;
END;
CREATE TRIGGER orch_stage_attempt_delete_guard BEFORE DELETE ON orch_stage_attempts
BEGIN SELECT RAISE(ABORT,'stage_attempt_delete_forbidden'); END;

CREATE TRIGGER orch_delivery_origin_guard BEFORE INSERT ON orch_delivery_obligations
BEGIN
  SELECT CASE WHEN orch_capability_ok(
    'delivery_obligation_insert',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,
    NEW.lifecycle_revision,NEW.cancel_epoch,NEW.delivery_key
  )!=1 THEN RAISE(ABORT,'missing_delivery_obligation_capability') END;
  SELECT CASE WHEN NEW.state!='pending' OR NOT EXISTS (
    SELECT 1 FROM orch_origins o
     WHERE o.board_instance_id=NEW.board_instance_id AND o.tenant_scope=NEW.tenant_scope
       AND o.origin_id=NEW.origin_id AND o.route_revision=NEW.route_revision
       AND o.route_digest=NEW.route_digest AND o.origin_kind=NEW.origin_kind
       AND o.platform=NEW.platform AND o.adapter_instance_id=NEW.adapter_instance_id
       AND o.account_id=NEW.account_id AND o.conversation_id=NEW.conversation_id
       AND o.thread_id=NEW.thread_id AND o.reply_to_id=NEW.reply_to_id
       AND o.required_ack_family=NEW.required_ack_family
       AND o.required_ack_strength=NEW.required_ack_strength
  ) THEN RAISE(ABORT,'delivery_origin_route_mismatch') END;
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM orch_requests r JOIN orch_delivery_manifests m
      ON m.board_instance_id=r.board_instance_id AND m.tenant_scope=r.tenant_scope
     AND m.orch_id=r.orch_id AND m.plan_version=NEW.plan_version
     AND m.result_id=NEW.result_id AND m.manifest_digest=NEW.manifest_digest
     WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope
       AND r.orch_id=NEW.orch_id AND r.plan_version=NEW.plan_version
       AND r.delivery_epoch_revision=NEW.lifecycle_revision AND r.cancel_epoch=NEW.cancel_epoch
       AND m.lifecycle_revision=NEW.lifecycle_revision AND m.cancel_epoch=NEW.cancel_epoch
       AND (
         (NEW.delivery_generation=1 AND NEW.supersedes_obligation_id IS NULL
          AND r.lifecycle_state='work_accepted')
         OR
         (NEW.delivery_generation>1 AND r.lifecycle_state='delivery_blocked'
          AND EXISTS (
            SELECT 1 FROM orch_delivery_obligations old
             WHERE old.board_instance_id=NEW.board_instance_id
               AND old.tenant_scope=NEW.tenant_scope
               AND old.obligation_id=NEW.supersedes_obligation_id
               AND old.orch_id=NEW.orch_id AND old.result_id=NEW.result_id
               AND old.manifest_entry_key=NEW.manifest_entry_key
               AND old.delivery_generation=NEW.delivery_generation-1
               AND (
                 old.state='dead_letter'
                 OR (old.state='unknown' AND EXISTS (
                   SELECT 1 FROM orch_delivery_resend_authorizations a
                    WHERE a.board_instance_id=NEW.board_instance_id
                      AND a.tenant_scope=NEW.tenant_scope
                      AND a.obligation_id=old.obligation_id
                      AND a.unknown_attempt_id=old.unknown_attempt_id
                      AND a.lifecycle_revision=NEW.lifecycle_revision
                      AND a.cancel_epoch=NEW.cancel_epoch
                      AND a.consumed_at IS NULL AND a.expires_at>unixepoch('now')
                 ))
               )
          ))
       )
  ) THEN RAISE(ABORT,'stale_or_unauthorized_delivery_binding') END;
END;

CREATE TRIGGER orch_delivery_identity_guard BEFORE UPDATE OF
  board_instance_id,tenant_scope,orch_id,plan_version,result_id,obligation_id,
  manifest_digest,manifest_entry_key,delivery_generation,supersedes_obligation_id,
  delivery_key,result_digest,origin_id,route_revision,route_digest,origin_kind,
  platform,adapter_instance_id,account_id,conversation_id,thread_id,reply_to_id,
  required,required_ack_family,required_ack_strength,lifecycle_revision,cancel_epoch,created_at
ON orch_delivery_obligations BEGIN SELECT RAISE(ABORT,'immutable_delivery_identity'); END;
CREATE TRIGGER orch_delivery_delete_guard BEFORE DELETE ON orch_delivery_obligations
BEGIN SELECT RAISE(ABORT,'delivery_delete_forbidden'); END;

CREATE TRIGGER orch_delivery_accept_guard BEFORE UPDATE OF state,acceptance_attempt_id ON orch_delivery_obligations
WHEN NEW.state='accepted'
BEGIN
  SELECT CASE WHEN OLD.state NOT IN ('claimed','unknown') OR NOT EXISTS (
    SELECT 1 FROM orch_delivery_attempts a JOIN orch_delivery_receipts rc
      ON rc.board_instance_id=a.board_instance_id AND rc.tenant_scope=a.tenant_scope
     AND rc.attempt_id=a.attempt_id AND rc.obligation_id=a.obligation_id
     WHERE a.board_instance_id=NEW.board_instance_id AND a.tenant_scope=NEW.tenant_scope
       AND a.obligation_id=NEW.obligation_id AND a.attempt_id=NEW.acceptance_attempt_id
       AND a.attempt_state='adapter_accepted' AND rc.verified=1
       AND a.claim_epoch=NEW.claim_epoch AND a.claim_owner=NEW.claim_owner
       AND a.claim_token_hash=NEW.claim_token_hash
       AND a.lifecycle_revision=NEW.lifecycle_revision AND a.cancel_epoch=NEW.cancel_epoch
       AND a.result_digest=NEW.result_digest AND a.route_digest=NEW.route_digest
       AND rc.send_nonce=a.send_nonce AND rc.payload_digest=a.payload_digest
       AND rc.result_digest=a.result_digest AND rc.route_digest=a.route_digest
       AND rc.observed_ack_family=NEW.required_ack_family
       AND rc.observed_ack_strength=NEW.required_ack_strength
  ) THEN RAISE(ABORT,'accepted_delivery_without_exact_receipt') END;
END;

CREATE TRIGGER orch_delivery_ack_guard BEFORE UPDATE OF state,acked_at ON orch_delivery_obligations
WHEN NEW.state='acked'
BEGIN
  SELECT CASE WHEN OLD.state NOT IN ('accepted','unknown') OR NOT EXISTS (
    SELECT 1 FROM orch_delivery_receipts rc
     WHERE rc.board_instance_id=NEW.board_instance_id AND rc.tenant_scope=NEW.tenant_scope
       AND rc.obligation_id=NEW.obligation_id AND rc.attempt_id=NEW.acceptance_attempt_id
       AND rc.verified=1 AND rc.observed_ack_family=NEW.required_ack_family
       AND rc.observed_ack_strength=NEW.required_ack_strength
       AND rc.result_digest=NEW.result_digest AND rc.route_digest=NEW.route_digest
  ) THEN RAISE(ABORT,'delivery_ack_family_or_strength_not_satisfied') END;
END;

CREATE TRIGGER orch_delivery_attempt_insert_guard BEFORE INSERT ON orch_delivery_attempts
BEGIN
  SELECT CASE WHEN NEW.attempt_state!='started' OR NEW.finished_at IS NOT NULL
    OR orch_capability_ok(
      'delivery_attempt_start',NEW.board_instance_id,NEW.tenant_scope,NEW.obligation_id,
      NEW.lifecycle_revision,NEW.cancel_epoch,NEW.payload_digest
    )!=1 OR NOT EXISTS (
      SELECT 1 FROM orch_delivery_obligations d JOIN orch_requests r
        ON r.board_instance_id=d.board_instance_id AND r.tenant_scope=d.tenant_scope AND r.orch_id=d.orch_id
       WHERE d.board_instance_id=NEW.board_instance_id AND d.tenant_scope=NEW.tenant_scope
         AND d.obligation_id=NEW.obligation_id AND d.state='claimed'
         AND d.claim_owner=NEW.claim_owner AND d.claim_token_hash=NEW.claim_token_hash
         AND d.claim_epoch=NEW.claim_epoch AND d.claim_expires_at>unixepoch('now')
         AND d.lifecycle_revision=NEW.lifecycle_revision AND d.cancel_epoch=NEW.cancel_epoch
         AND d.result_digest=NEW.result_digest AND d.route_digest=NEW.route_digest
         AND r.delivery_epoch_revision=NEW.lifecycle_revision AND r.cancel_epoch=NEW.cancel_epoch
         AND r.lifecycle_state IN ('delivering','delivery_blocked')
    ) THEN RAISE(ABORT,'invalid_delivery_attempt_claim') END;
END;

CREATE TRIGGER orch_delivery_attempt_transition_guard BEFORE UPDATE ON orch_delivery_attempts
BEGIN
  SELECT CASE WHEN NOT (
    OLD.attempt_state='started' AND NEW.attempt_state IN ('adapter_accepted','rejected','unknown')
    AND NEW.finished_at IS NOT NULL
    AND NEW.board_instance_id=OLD.board_instance_id AND NEW.tenant_scope=OLD.tenant_scope
    AND NEW.obligation_id=OLD.obligation_id AND NEW.attempt_id=OLD.attempt_id
    AND NEW.claim_epoch=OLD.claim_epoch AND NEW.claim_owner=OLD.claim_owner
    AND NEW.claim_token_hash=OLD.claim_token_hash
    AND NEW.lifecycle_revision=OLD.lifecycle_revision AND NEW.cancel_epoch=OLD.cancel_epoch
    AND NEW.send_nonce=OLD.send_nonce AND NEW.payload_digest=OLD.payload_digest
    AND NEW.result_digest=OLD.result_digest AND NEW.route_digest=OLD.route_digest
    AND NEW.started_at=OLD.started_at
    AND orch_capability_ok(
      'delivery_attempt_finish',NEW.board_instance_id,NEW.tenant_scope,NEW.obligation_id,
      NEW.lifecycle_revision,NEW.cancel_epoch,NEW.payload_digest
    )=1
    AND EXISTS (
      SELECT 1 FROM orch_delivery_obligations d JOIN orch_requests r
        ON r.board_instance_id=d.board_instance_id AND r.tenant_scope=d.tenant_scope AND r.orch_id=d.orch_id
       WHERE d.board_instance_id=NEW.board_instance_id AND d.tenant_scope=NEW.tenant_scope
         AND d.obligation_id=NEW.obligation_id AND d.claim_epoch=NEW.claim_epoch
         AND d.claim_owner=NEW.claim_owner AND d.claim_token_hash=NEW.claim_token_hash
         AND d.lifecycle_revision=NEW.lifecycle_revision AND d.cancel_epoch=NEW.cancel_epoch
         AND r.delivery_epoch_revision=NEW.lifecycle_revision
         AND ((r.cancel_epoch=NEW.cancel_epoch
               AND r.lifecycle_state IN ('delivering','delivery_blocked'))
           OR (r.cancel_epoch=NEW.cancel_epoch+1 AND r.lifecycle_state='cancelling'))
    )
  ) THEN RAISE(ABORT,'invalid_delivery_attempt_transition') END;
END;

CREATE TRIGGER orch_delivery_receipt_insert_guard BEFORE INSERT ON orch_delivery_receipts
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM orch_delivery_attempts a JOIN orch_delivery_obligations d
      ON d.board_instance_id=a.board_instance_id AND d.tenant_scope=a.tenant_scope
     AND d.obligation_id=a.obligation_id
     WHERE a.board_instance_id=NEW.board_instance_id AND a.tenant_scope=NEW.tenant_scope
       AND a.obligation_id=NEW.obligation_id AND a.attempt_id=NEW.attempt_id
       AND a.attempt_state IN ('adapter_accepted','unknown')
       AND NEW.send_nonce=a.send_nonce AND NEW.payload_digest=a.payload_digest
       AND NEW.result_digest=a.result_digest AND NEW.route_digest=a.route_digest
       AND NEW.observed_ack_family=d.required_ack_family
       AND NEW.observed_ack_strength=d.required_ack_strength
       AND orch_capability_ok(
         'delivery_receipt',NEW.board_instance_id,NEW.tenant_scope,NEW.obligation_id,
         a.lifecycle_revision,a.cancel_epoch,NEW.receipt_digest
       )=1
  ) THEN RAISE(ABORT,'receipt_without_exact_attempt_acceptance') END;
END;

CREATE TRIGGER orch_delivery_attempt_event_guard BEFORE INSERT ON orch_delivery_attempt_events
BEGIN
  SELECT CASE WHEN NEW.event_kind!=NEW.to_state OR NOT EXISTS (
    SELECT 1 FROM orch_delivery_attempts a
     WHERE a.board_instance_id=NEW.board_instance_id AND a.tenant_scope=NEW.tenant_scope
       AND a.attempt_id=NEW.attempt_id AND a.attempt_state=NEW.to_state
       AND NEW.transition_seq IN (1,2)
       AND ((NEW.transition_seq=1 AND NEW.from_state IS NULL AND NEW.to_state='started')
         OR (NEW.transition_seq=2 AND NEW.from_state='started' AND NEW.to_state!='started'))
  ) THEN RAISE(ABORT,'invalid_delivery_attempt_event') END;
END;
CREATE TRIGGER orch_delivery_attempt_delete_guard BEFORE DELETE ON orch_delivery_attempts
BEGIN SELECT RAISE(ABORT,'delivery_attempt_delete_forbidden'); END;

CREATE TRIGGER tasks_orch_runtime_write_guard BEFORE UPDATE ON tasks
WHEN OLD.orch_id IS NOT NULL
  AND NEW.orch_board_instance_id IS OLD.orch_board_instance_id
  AND NEW.orch_tenant_scope IS OLD.orch_tenant_scope
  AND NEW.orch_id IS OLD.orch_id
  AND NEW.orch_plan_version IS OLD.orch_plan_version
  AND NEW.orch_node_key IS OLD.orch_node_key
  AND NEW.orch_binding_revision IS OLD.orch_binding_revision
  AND NEW.orch_cancel_epoch IS OLD.orch_cancel_epoch
BEGIN
  SELECT CASE WHEN orch_capability_ok(
    'task_write',OLD.orch_board_instance_id,OLD.orch_tenant_scope,OLD.orch_id,
    COALESCE((SELECT lifecycle_revision FROM orch_requests r
      WHERE r.board_instance_id=OLD.orch_board_instance_id AND r.tenant_scope=OLD.orch_tenant_scope
        AND r.orch_id=OLD.orch_id),-1),
    COALESCE((SELECT cancel_epoch FROM orch_requests r
      WHERE r.board_instance_id=OLD.orch_board_instance_id AND r.tenant_scope=OLD.orch_tenant_scope
        AND r.orch_id=OLD.orch_id),-1),OLD.id
  )!=1 THEN RAISE(ABORT,'missing_orch_task_write_capability') END;
END;
CREATE TRIGGER tasks_orch_id_update_guard BEFORE UPDATE OF id ON tasks
WHEN OLD.orch_id IS NOT NULL
BEGIN SELECT RAISE(ABORT,'orch_task_id_immutable'); END;
CREATE TRIGGER tasks_orch_delete_guard BEFORE DELETE ON tasks
WHEN OLD.orch_id IS NOT NULL
BEGIN SELECT RAISE(ABORT,'orch_bound_task_delete_forbidden'); END;

CREATE TRIGGER task_runs_orch_insert_guard BEFORE INSERT ON task_runs
WHEN EXISTS (SELECT 1 FROM tasks t WHERE t.id=NEW.task_id AND t.orch_id IS NOT NULL)
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM tasks t JOIN orch_requests r
      ON r.board_instance_id=t.orch_board_instance_id
     AND r.tenant_scope=t.orch_tenant_scope AND r.orch_id=t.orch_id
     WHERE t.id=NEW.task_id AND NEW.cancellation_epoch=r.cancel_epoch
       AND r.lifecycle_state NOT IN ('cancelling','completed','failed','cancelled')
       AND orch_capability_ok(
         'task_run_start',r.board_instance_id,r.tenant_scope,r.orch_id,
         r.lifecycle_revision,r.cancel_epoch,COALESCE(CAST(NEW.id AS TEXT),NEW.task_id)
       )=1
  ) THEN RAISE(ABORT,'invalid_orch_task_run_start') END;
END;
CREATE TRIGGER task_runs_orch_update_guard BEFORE UPDATE ON task_runs
WHEN EXISTS (SELECT 1 FROM tasks t WHERE t.id IN (OLD.task_id,NEW.task_id) AND t.orch_id IS NOT NULL)
BEGIN
  SELECT CASE WHEN NEW.id!=OLD.id OR NEW.task_id!=OLD.task_id OR NOT EXISTS (
    SELECT 1 FROM tasks t JOIN orch_requests r
      ON r.board_instance_id=t.orch_board_instance_id
     AND r.tenant_scope=t.orch_tenant_scope AND r.orch_id=t.orch_id
     WHERE t.id=NEW.task_id AND NEW.cancellation_epoch=r.cancel_epoch
       AND r.lifecycle_state NOT IN ('completed','failed','cancelled')
       AND orch_capability_ok(
         'task_run_write',r.board_instance_id,r.tenant_scope,r.orch_id,
         r.lifecycle_revision,r.cancel_epoch,CAST(NEW.id AS TEXT)
       )=1
  ) THEN RAISE(ABORT,'invalid_orch_task_run_write') END;
END;
CREATE TRIGGER task_runs_orch_delete_guard BEFORE DELETE ON task_runs
WHEN EXISTS (SELECT 1 FROM tasks t WHERE t.id=OLD.task_id AND t.orch_id IS NOT NULL)
BEGIN SELECT RAISE(ABORT,'orch_task_run_delete_forbidden'); END;

CREATE TRIGGER task_links_orch_update_guard BEFORE UPDATE ON task_links
WHEN OLD.orch_id IS NOT NULL OR NEW.orch_id IS NOT NULL
BEGIN SELECT RAISE(ABORT,'replace_orch_link_via_typed_delete_insert'); END;
CREATE TRIGGER task_links_orch_delete_guard BEFORE DELETE ON task_links
WHEN OLD.orch_id IS NOT NULL
BEGIN
  SELECT CASE WHEN NOT (
    EXISTS (
      SELECT 1 FROM orch_requests r
       WHERE r.board_instance_id=OLD.orch_board_instance_id
         AND r.tenant_scope=OLD.orch_tenant_scope AND r.orch_id=OLD.orch_id
         AND r.lifecycle_state IN ('cancelling','cancelled')
    )
    AND orch_capability_ok(
      'link_retire',OLD.orch_board_instance_id,OLD.orch_tenant_scope,OLD.orch_id,
      COALESCE((SELECT lifecycle_revision FROM orch_requests r
        WHERE r.board_instance_id=OLD.orch_board_instance_id AND r.tenant_scope=OLD.orch_tenant_scope
          AND r.orch_id=OLD.orch_id),-1),
      COALESCE((SELECT cancel_epoch FROM orch_requests r
        WHERE r.board_instance_id=OLD.orch_board_instance_id AND r.tenant_scope=OLD.orch_tenant_scope
          AND r.orch_id=OLD.orch_id),-1),OLD.orch_edge_key
    )=1
  ) THEN RAISE(ABORT,'orch_link_delete_without_retirement_authority') END;
END;

CREATE TRIGGER board_identity_insert_guard BEFORE INSERT ON kanban_board_identity BEGIN
  SELECT CASE WHEN orch_capability_ok('maintenance_identity',NEW.board_instance_id,'','',0,0,NEW.canonical_board_key)!=1
    THEN RAISE(ABORT,'missing_maintenance_identity_capability') END;
END;
CREATE TRIGGER board_identity_update_guard BEFORE UPDATE ON kanban_board_identity BEGIN SELECT RAISE(ABORT,'board_identity_immutable'); END;
CREATE TRIGGER board_identity_delete_guard BEFORE DELETE ON kanban_board_identity BEGIN SELECT RAISE(ABORT,'board_identity_delete_forbidden'); END;

CREATE TRIGGER schema_migration_insert_guard BEFORE INSERT ON kanban_schema_migrations BEGIN
  SELECT CASE WHEN orch_capability_ok('maintenance_schema',NEW.board_instance_id,'',NEW.migration_id,NEW.fence_generation,0,NEW.source_digest)!=1
    THEN RAISE(ABORT,'missing_schema_migration_capability') END;
END;
CREATE TRIGGER schema_migration_update_guard BEFORE UPDATE ON kanban_schema_migrations BEGIN
  SELECT CASE WHEN NEW.migration_id!=OLD.migration_id OR NEW.board_instance_id!=OLD.board_instance_id
    OR NEW.target_schema_version!=OLD.target_schema_version OR NEW.source_digest!=OLD.source_digest
    OR NEW.backup_digest!=OLD.backup_digest OR NEW.fence_generation!=OLD.fence_generation
    OR orch_capability_ok('maintenance_schema',NEW.board_instance_id,'',NEW.migration_id,NEW.fence_generation,0,NEW.state)!=1
    THEN RAISE(ABORT,'invalid_schema_migration_transition') END;
END;
CREATE TRIGGER schema_migration_delete_guard BEFORE DELETE ON kanban_schema_migrations BEGIN SELECT RAISE(ABORT,'schema_migration_delete_forbidden'); END;

CREATE TRIGGER write_fence_insert_guard BEFORE INSERT ON kanban_write_fence BEGIN
  SELECT CASE WHEN orch_capability_ok('maintenance_fence',(SELECT board_instance_id FROM kanban_board_identity WHERE singleton=1),'','',NEW.generation,0,NEW.mode)!=1
    THEN RAISE(ABORT,'missing_write_fence_capability') END;
END;
CREATE TRIGGER write_fence_update_guard BEFORE UPDATE ON kanban_write_fence BEGIN
  SELECT CASE WHEN NEW.singleton!=OLD.singleton OR NEW.generation!=OLD.generation+1
    OR orch_capability_ok('maintenance_fence',(SELECT board_instance_id FROM kanban_board_identity WHERE singleton=1),'','',NEW.generation,0,NEW.mode)!=1
    THEN RAISE(ABORT,'invalid_write_fence_transition') END;
END;
CREATE TRIGGER write_fence_delete_guard BEFORE DELETE ON kanban_write_fence BEGIN SELECT RAISE(ABORT,'write_fence_delete_forbidden'); END;

CREATE TRIGGER commit_clock_insert_guard BEFORE INSERT ON kanban_commit_clock BEGIN
  SELECT CASE WHEN NEW.commit_seq!=0 OR orch_capability_ok('commit_clock',(SELECT board_instance_id FROM kanban_board_identity WHERE singleton=1),'','',0,0,NEW.last_txn_id)!=1
    THEN RAISE(ABORT,'invalid_commit_clock_bootstrap') END;
END;
CREATE TRIGGER commit_clock_update_guard BEFORE UPDATE ON kanban_commit_clock BEGIN
  SELECT CASE WHEN NEW.singleton!=OLD.singleton OR NEW.commit_seq!=OLD.commit_seq+1
    OR orch_capability_ok('commit_clock',(SELECT board_instance_id FROM kanban_board_identity WHERE singleton=1),'','',NEW.commit_seq,0,NEW.last_txn_id)!=1
    THEN RAISE(ABORT,'invalid_commit_clock_cas') END;
END;
CREATE TRIGGER commit_clock_delete_guard BEFORE DELETE ON kanban_commit_clock BEGIN SELECT RAISE(ABORT,'commit_clock_delete_forbidden'); END;

CREATE TRIGGER migration_operation_insert_guard BEFORE INSERT ON kanban_migration_operations BEGIN
  SELECT CASE WHEN orch_capability_ok('maintenance_operation',NEW.board_instance_id,'',NEW.migration_id,NEW.phase_revision,NEW.fence_generation,NEW.plan_digest)!=1
    THEN RAISE(ABORT,'missing_migration_operation_capability') END;
END;
CREATE TRIGGER migration_operation_update_guard BEFORE UPDATE ON kanban_migration_operations BEGIN
  SELECT CASE WHEN NEW.migration_id!=OLD.migration_id OR NEW.board_instance_id!=OLD.board_instance_id
    OR NEW.owner_token_hash!=OLD.owner_token_hash OR NEW.source_digest!=OLD.source_digest
    OR NEW.plan_digest!=OLD.plan_digest OR NEW.schema_digest!=OLD.schema_digest
    OR NEW.phase_revision!=OLD.phase_revision+1
    OR orch_capability_ok('maintenance_operation',NEW.board_instance_id,'',NEW.migration_id,NEW.phase_revision,NEW.fence_generation,NEW.phase)!=1
    THEN RAISE(ABORT,'invalid_migration_operation_transition') END;
END;
CREATE TRIGGER migration_operation_delete_guard BEFORE DELETE ON kanban_migration_operations BEGIN SELECT RAISE(ABORT,'migration_operation_delete_forbidden'); END;

CREATE TRIGGER rollback_operation_insert_guard BEFORE INSERT ON orch_rollback_operations BEGIN
  SELECT CASE WHEN orch_capability_ok('rollback_operation',NEW.board_instance_id,'',NEW.rollback_id,NEW.phase_revision,NEW.fence_generation,NEW.target_manifest_digest)!=1
    THEN RAISE(ABORT,'missing_rollback_operation_capability') END;
END;
CREATE TRIGGER rollback_operation_update_guard BEFORE UPDATE ON orch_rollback_operations BEGIN
  SELECT CASE WHEN NEW.rollback_id!=OLD.rollback_id OR NEW.migration_id!=OLD.migration_id
    OR NEW.board_instance_id!=OLD.board_instance_id OR NEW.owner_token_hash!=OLD.owner_token_hash
    OR NEW.source_manifest_digest!=OLD.source_manifest_digest OR NEW.target_manifest_digest!=OLD.target_manifest_digest
    OR NEW.phase_revision!=OLD.phase_revision+1
    OR orch_capability_ok('rollback_operation',NEW.board_instance_id,'',NEW.rollback_id,NEW.phase_revision,NEW.fence_generation,NEW.phase)!=1
    THEN RAISE(ABORT,'invalid_rollback_operation_transition') END;
END;
CREATE TRIGGER rollback_operation_delete_guard BEFORE DELETE ON orch_rollback_operations BEGIN SELECT RAISE(ABORT,'rollback_operation_delete_forbidden'); END;

CREATE TRIGGER replay_selector_insert_guard BEFORE INSERT ON orch_replay_selectors BEGIN
  SELECT CASE WHEN NEW.current_generation!=0 OR NEW.current_orch_id IS NOT NULL OR NEW.current_request_digest IS NOT NULL
    OR NEW.ledger_revision!=0
    OR orch_capability_ok('selector_create',NEW.board_instance_id,NEW.tenant_scope,NEW.lineage_id,0,0,NEW.selector_key)!=1
    THEN RAISE(ABORT,'invalid_replay_selector_create') END;
END;
CREATE TRIGGER replay_selector_update_guard BEFORE UPDATE ON orch_replay_selectors BEGIN
  SELECT CASE WHEN NEW.board_instance_id!=OLD.board_instance_id OR NEW.tenant_scope!=OLD.tenant_scope
    OR NEW.selector_key!=OLD.selector_key OR NEW.selector_kind!=OLD.selector_kind
    OR NEW.selector_value!=OLD.selector_value OR NEW.adapter_instance_id!=OLD.adapter_instance_id
    OR NEW.conversation_id!=OLD.conversation_id OR NEW.lineage_id!=OLD.lineage_id
    OR NEW.created_at!=OLD.created_at OR NEW.current_generation!=OLD.current_generation+1
    OR NEW.ledger_revision!=OLD.ledger_revision+1
    OR orch_capability_ok('selector_advance',NEW.board_instance_id,NEW.tenant_scope,NEW.lineage_id,OLD.ledger_revision,NEW.current_generation,NEW.selector_key)!=1
    OR NOT EXISTS (SELECT 1 FROM orch_requests r
      WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope
        AND r.orch_id=NEW.current_orch_id AND r.generation=NEW.current_generation
        AND r.request_digest=NEW.current_request_digest AND r.selector_key=NEW.selector_key)
    THEN RAISE(ABORT,'invalid_replay_selector_advance') END;
END;
CREATE TRIGGER replay_selector_delete_guard BEFORE DELETE ON orch_replay_selectors BEGIN SELECT RAISE(ABORT,'replay_selector_delete_forbidden'); END;

CREATE TRIGGER orch_origin_insert_guard BEFORE INSERT ON orch_origins BEGIN
  SELECT CASE WHEN orch_capability_ok('origin_register',NEW.board_instance_id,NEW.tenant_scope,NEW.origin_id,NEW.route_revision,0,NEW.route_digest)!=1
    THEN RAISE(ABORT,'missing_origin_register_capability') END;
END;

CREATE TRIGGER orch_request_transition_guard BEFORE UPDATE ON orch_requests BEGIN
  SELECT CASE WHEN NEW.lifecycle_revision!=OLD.lifecycle_revision+1 OR NEW.updated_at<OLD.updated_at
    OR orch_capability_ok('request_transition',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,NEW.lifecycle_revision,NEW.cancel_epoch,NEW.request_key)!=1
    OR NOT (
      (OLD.lifecycle_state='submitted' AND NEW.lifecycle_state='decomposing')
      OR (OLD.lifecycle_state='decomposing' AND NEW.lifecycle_state IN ('waiting_lanes','blocked','failed','cancelling'))
      OR (OLD.lifecycle_state='waiting_lanes' AND NEW.lifecycle_state IN ('waiting_lanes','synthesizing','blocked','failed','cancelling'))
      OR (OLD.lifecycle_state='synthesizing' AND NEW.lifecycle_state IN ('work_accepted','blocked','failed','cancelling'))
      OR (OLD.lifecycle_state='work_accepted' AND NEW.lifecycle_state IN ('delivering','completed','cancelling'))
      OR (OLD.lifecycle_state='delivering' AND NEW.lifecycle_state IN ('completed','delivery_blocked','cancelling'))
      OR (OLD.lifecycle_state='delivery_blocked' AND NEW.lifecycle_state IN ('delivering','cancelling'))
      OR (OLD.lifecycle_state='blocked' AND NEW.lifecycle_state=OLD.resume_state)
      OR (OLD.lifecycle_state='cancelling' AND NEW.lifecycle_state='cancelled')
    )
    OR NOT ((NEW.lifecycle_state='cancelling' AND OLD.lifecycle_state!='cancelling' AND NEW.cancel_epoch=OLD.cancel_epoch+1)
      OR (NEW.lifecycle_state!='cancelling' AND NEW.cancel_epoch=OLD.cancel_epoch))
    OR NOT ((OLD.lifecycle_state='synthesizing' AND NEW.lifecycle_state='work_accepted'
              AND OLD.delivery_epoch_revision=0 AND NEW.delivery_epoch_revision=NEW.lifecycle_revision
              AND NEW.work_accepted_at IS NOT NULL)
      OR (NOT (OLD.lifecycle_state='synthesizing' AND NEW.lifecycle_state='work_accepted')
              AND NEW.delivery_epoch_revision=OLD.delivery_epoch_revision
              AND NEW.work_accepted_at IS OLD.work_accepted_at))
    OR NOT ((OLD.lifecycle_state='decomposing' AND NEW.lifecycle_state='waiting_lanes'
              AND NEW.plan_version=OLD.plan_version+1
              AND OLD.plan_epoch_revision=0 AND NEW.plan_epoch_revision=NEW.lifecycle_revision
              AND EXISTS (SELECT 1 FROM orch_plan_materializations m
                WHERE m.board_instance_id=NEW.board_instance_id AND m.tenant_scope=NEW.tenant_scope
                  AND m.orch_id=NEW.orch_id AND m.plan_version=NEW.plan_version
                  AND m.request_lifecycle_revision=OLD.lifecycle_revision
                  AND m.cancel_epoch=OLD.cancel_epoch))
      OR (NOT (OLD.lifecycle_state='decomposing' AND NEW.lifecycle_state='waiting_lanes')
              AND NEW.plan_version=OLD.plan_version
              AND NEW.plan_epoch_revision=OLD.plan_epoch_revision))
    OR (NEW.lifecycle_state='completed' AND NEW.delivery_closed_at IS NULL)
    OR (NEW.lifecycle_state!='completed' AND NEW.delivery_closed_at IS NOT OLD.delivery_closed_at)
    THEN RAISE(ABORT,'invalid_orch_request_transition') END;
END;

CREATE TRIGGER orch_requirement_insert_guard BEFORE INSERT ON orch_request_requirements BEGIN
  SELECT CASE WHEN orch_capability_ok('plan_build',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,
      COALESCE((SELECT lifecycle_revision FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),
      COALESCE((SELECT cancel_epoch FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),NEW.requirement_digest)!=1
    THEN RAISE(ABORT,'missing_plan_build_capability') END;
END;
CREATE TRIGGER orch_plan_insert_guard BEFORE INSERT ON orch_plans BEGIN
  SELECT CASE WHEN orch_capability_ok('plan_build',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,
      COALESCE((SELECT lifecycle_revision FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),
      COALESCE((SELECT cancel_epoch FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),NEW.plan_digest)!=1
    THEN RAISE(ABORT,'missing_plan_build_capability') END;
END;
CREATE TRIGGER orch_plan_node_insert_guard BEFORE INSERT ON orch_plan_nodes BEGIN
  SELECT CASE WHEN orch_capability_ok('plan_build',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,
      COALESCE((SELECT lifecycle_revision FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),
      COALESCE((SELECT cancel_epoch FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),NEW.node_key)!=1
    THEN RAISE(ABORT,'missing_plan_node_capability') END;
END;
CREATE TRIGGER orch_plan_edge_insert_guard BEFORE INSERT ON orch_plan_edges BEGIN
  SELECT CASE WHEN orch_capability_ok('plan_build',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,
      COALESCE((SELECT lifecycle_revision FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),
      COALESCE((SELECT cancel_epoch FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),NEW.edge_key)!=1
    THEN RAISE(ABORT,'missing_plan_edge_capability') END;
END;
CREATE TRIGGER orch_plan_coverage_insert_guard BEFORE INSERT ON orch_plan_coverage BEGIN
  SELECT CASE WHEN orch_capability_ok('plan_build',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,
      COALESCE((SELECT lifecycle_revision FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),
      COALESCE((SELECT cancel_epoch FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),NEW.requirement_id||':'||NEW.node_key)!=1
    THEN RAISE(ABORT,'missing_plan_coverage_capability') END;
END;

CREATE TRIGGER orch_node_insert_guard BEFORE INSERT ON orch_nodes BEGIN
  SELECT CASE WHEN orch_capability_ok('materialize',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,
      COALESCE((SELECT lifecycle_revision FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),
      COALESCE((SELECT cancel_epoch FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),NEW.node_key)!=1
    THEN RAISE(ABORT,'missing_node_materialization_capability') END;
END;
CREATE TRIGGER orch_node_state_guard BEFORE UPDATE OF node_state,updated_at ON orch_nodes BEGIN
  SELECT CASE WHEN NEW.board_instance_id!=OLD.board_instance_id OR NEW.tenant_scope!=OLD.tenant_scope
    OR NEW.orch_id!=OLD.orch_id OR NEW.plan_version!=OLD.plan_version OR NEW.node_key!=OLD.node_key
    OR NEW.task_id!=OLD.task_id OR NEW.updated_at<OLD.updated_at
    OR NOT ((OLD.node_state='planned' AND NEW.node_state IN ('ready','running','cancelled'))
      OR (OLD.node_state='ready' AND NEW.node_state IN ('running','cancelled'))
      OR (OLD.node_state='running' AND NEW.node_state IN ('accepted','blocked','failed','cancellation_requested','cancelled'))
      OR (OLD.node_state='cancellation_requested' AND NEW.node_state IN ('accepted','failed','cancelled'))
      OR (OLD.node_state='blocked' AND NEW.node_state IN ('ready','cancelled')))
    OR orch_capability_ok('node_transition',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,
      COALESCE((SELECT lifecycle_revision FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),
      COALESCE((SELECT cancel_epoch FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),NEW.node_key)!=1
    THEN RAISE(ABORT,'invalid_orch_node_transition') END;
END;

CREATE TRIGGER orch_external_edge_insert_guard BEFORE INSERT ON orch_external_edges BEGIN
  SELECT CASE WHEN orch_capability_ok('external_edge',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,NEW.lifecycle_revision,
      COALESCE((SELECT cancel_epoch FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),NEW.edge_key)!=1
    THEN RAISE(ABORT,'missing_external_edge_capability') END;
END;
CREATE TRIGGER orch_external_edge_update_guard BEFORE UPDATE ON orch_external_edges BEGIN SELECT RAISE(ABORT,'immutable_external_edge'); END;
CREATE TRIGGER orch_external_edge_delete_guard BEFORE DELETE ON orch_external_edges BEGIN SELECT RAISE(ABORT,'external_edge_delete_forbidden'); END;

CREATE TRIGGER orch_stage_lease_insert_guard BEFORE INSERT ON orch_stage_leases BEGIN
  SELECT CASE WHEN NEW.lease_state!='active' OR NEW.epoch!=1 OR NEW.expires_at<=unixepoch('now')
    OR orch_capability_ok('lease_claim',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,
      COALESCE((SELECT lifecycle_revision FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),
      COALESCE((SELECT cancel_epoch FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),NEW.stage)!=1
    OR NOT EXISTS (SELECT 1 FROM orch_requests r
      WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id
        AND ((NEW.stage='decomposition' AND r.lifecycle_state='decomposing')
          OR (NEW.stage='synthesis' AND r.lifecycle_state='synthesizing')
          OR (NEW.stage='reconciliation' AND r.lifecycle_state IN ('waiting_lanes','delivering','delivery_blocked','cancelling'))))
    THEN RAISE(ABORT,'invalid_stage_lease_claim') END;
END;
CREATE TRIGGER orch_stage_lease_update_guard BEFORE UPDATE ON orch_stage_leases BEGIN
  SELECT CASE WHEN NEW.board_instance_id!=OLD.board_instance_id OR NEW.tenant_scope!=OLD.tenant_scope
    OR NEW.orch_id!=OLD.orch_id OR NEW.stage!=OLD.stage OR NEW.updated_at<OLD.updated_at
    OR orch_capability_ok('lease_update',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,
      COALESCE((SELECT lifecycle_revision FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),
      COALESCE((SELECT cancel_epoch FROM orch_requests r WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id),-1),NEW.stage)!=1
    OR NOT (
      (OLD.lease_state='active' AND NEW.lease_state='active' AND NEW.owner_run_id=OLD.owner_run_id
        AND NEW.owner_profile=OLD.owner_profile AND NEW.token_hash=OLD.token_hash
        AND NEW.epoch=OLD.epoch AND NEW.expires_at>OLD.expires_at)
      OR (OLD.lease_state='active' AND NEW.lease_state IN ('released','revoked')
        AND NEW.owner_run_id=OLD.owner_run_id AND NEW.owner_profile=OLD.owner_profile
        AND NEW.token_hash=OLD.token_hash AND NEW.epoch=OLD.epoch)
      OR (NEW.lease_state='active' AND NEW.epoch=OLD.epoch+1 AND NEW.expires_at>unixepoch('now')
        AND (OLD.lease_state IN ('released','revoked') OR OLD.expires_at<=unixepoch('now')))
    )
    THEN RAISE(ABORT,'invalid_stage_lease_transition') END;
END;
CREATE TRIGGER orch_stage_lease_delete_guard BEFORE DELETE ON orch_stage_leases BEGIN SELECT RAISE(ABORT,'stage_lease_delete_forbidden'); END;

CREATE TRIGGER orch_result_insert_guard BEFORE INSERT ON orch_results BEGIN
  SELECT CASE WHEN orch_capability_ok('result_accept',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,NEW.lifecycle_revision,NEW.cancel_epoch,NEW.result_digest)!=1
    OR NOT EXISTS (SELECT 1 FROM orch_requests r JOIN orch_stage_attempts a
      ON a.board_instance_id=r.board_instance_id AND a.tenant_scope=r.tenant_scope AND a.orch_id=r.orch_id
      WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id
        AND r.plan_version=NEW.plan_version AND r.lifecycle_state='synthesizing'
        AND r.lifecycle_revision=NEW.lifecycle_revision AND r.cancel_epoch=NEW.cancel_epoch
        AND a.stage=NEW.producer_stage AND a.attempt_no=NEW.producer_attempt_no
        AND a.owner_run_id=NEW.producer_run_id AND a.lease_epoch=NEW.synthesis_epoch
        AND a.lifecycle_revision=NEW.lifecycle_revision AND a.cancel_epoch=NEW.cancel_epoch
        AND a.attempt_state='completed' AND a.outcome_code='result_ready')
    THEN RAISE(ABORT,'invalid_result_acceptance') END;
END;

CREATE TRIGGER delivery_manifest_insert_guard BEFORE INSERT ON orch_delivery_manifests BEGIN
  SELECT CASE WHEN orch_capability_ok('delivery_manifest',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,NEW.lifecycle_revision,NEW.cancel_epoch,NEW.manifest_digest)!=1
    OR NOT EXISTS (SELECT 1 FROM orch_requests r JOIN orch_results rs
      ON rs.board_instance_id=r.board_instance_id AND rs.tenant_scope=r.tenant_scope
      AND rs.orch_id=r.orch_id AND rs.plan_version=r.plan_version
      WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id
        AND r.plan_version=NEW.plan_version AND r.lifecycle_state='work_accepted'
        AND r.delivery_epoch_revision=NEW.lifecycle_revision AND r.cancel_epoch=NEW.cancel_epoch
        AND rs.result_id=NEW.result_id)
    THEN RAISE(ABORT,'invalid_delivery_manifest') END;
END;
CREATE TRIGGER delivery_manifest_update_guard BEFORE UPDATE ON orch_delivery_manifests BEGIN SELECT RAISE(ABORT,'immutable_delivery_manifest'); END;
CREATE TRIGGER delivery_manifest_delete_guard BEFORE DELETE ON orch_delivery_manifests BEGIN SELECT RAISE(ABORT,'delivery_manifest_delete_forbidden'); END;
CREATE TRIGGER delivery_manifest_entry_insert_guard BEFORE INSERT ON orch_delivery_manifest_entries BEGIN
  SELECT CASE WHEN orch_capability_ok('delivery_manifest_entry',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,0,0,NEW.manifest_entry_key)!=1
    OR NOT EXISTS (SELECT 1 FROM orch_delivery_obligations d
      WHERE d.board_instance_id=NEW.board_instance_id AND d.tenant_scope=NEW.tenant_scope
        AND d.obligation_id=NEW.obligation_id AND d.orch_id=NEW.orch_id AND d.result_id=NEW.result_id
        AND d.manifest_digest=NEW.manifest_digest AND d.manifest_entry_key=NEW.manifest_entry_key
        AND d.required=NEW.required AND d.required_ack_family=NEW.required_ack_family
        AND d.required_ack_strength=NEW.required_ack_strength AND d.route_digest=NEW.route_digest)
    THEN RAISE(ABORT,'invalid_delivery_manifest_entry') END;
END;
CREATE TRIGGER delivery_manifest_entry_update_guard BEFORE UPDATE ON orch_delivery_manifest_entries BEGIN SELECT RAISE(ABORT,'immutable_delivery_manifest_entry'); END;
CREATE TRIGGER delivery_manifest_entry_delete_guard BEFORE DELETE ON orch_delivery_manifest_entries BEGIN SELECT RAISE(ABORT,'delivery_manifest_entry_delete_forbidden'); END;

CREATE TRIGGER orch_delivery_state_guard BEFORE UPDATE OF state,claim_owner,claim_token_hash,claim_epoch,claim_expires_at,
  attempts,available_at,acceptance_attempt_id,adapter_accepted_at,acked_at,provider_id,provider_message_id,
  duplicate_possible,unknown_attempt_id,ambiguity_deadline,last_error_code,updated_at ON orch_delivery_obligations
BEGIN
  SELECT CASE WHEN orch_capability_ok(
      CASE
        WHEN OLD.state='pending' AND NEW.state='claimed' THEN 'delivery_claim'
        WHEN NEW.state IN ('accepted','pending','unknown','dead_letter') THEN 'delivery_outcome'
        WHEN NEW.state='acked' THEN 'delivery_ack'
        WHEN NEW.state='cancelled' THEN 'delivery_cancel'
        ELSE 'delivery_invalid'
      END,
      NEW.board_instance_id,NEW.tenant_scope,NEW.obligation_id,
      NEW.lifecycle_revision,NEW.cancel_epoch,NEW.delivery_key
    )!=1
    OR NOT ((OLD.state='pending' AND NEW.state IN ('claimed','cancelled'))
      OR (OLD.state='claimed' AND NEW.state IN ('accepted','pending','unknown','dead_letter','cancelled'))
      OR (OLD.state='unknown' AND NEW.state IN ('accepted','acked','cancelled'))
      OR (OLD.state='accepted' AND NEW.state IN ('acked','cancelled')))
    OR (NEW.state='claimed' AND (NEW.claim_owner IS NULL OR NEW.claim_token_hash IS NULL
      OR NEW.claim_epoch!=OLD.claim_epoch+1 OR NEW.claim_expires_at<=unixepoch('now') OR NEW.attempts!=OLD.attempts+1))
    OR (NEW.state!='claimed' AND NEW.state!='accepted'
      AND (NEW.claim_owner IS NOT NULL OR NEW.claim_token_hash IS NOT NULL OR NEW.claim_expires_at IS NOT NULL))
    OR (NEW.state='unknown' AND (NEW.unknown_attempt_id IS NULL OR NEW.ambiguity_deadline IS NULL))
    OR NEW.updated_at<OLD.updated_at
    OR NOT EXISTS (SELECT 1 FROM orch_requests r
      WHERE r.board_instance_id=NEW.board_instance_id AND r.tenant_scope=NEW.tenant_scope AND r.orch_id=NEW.orch_id
        AND r.delivery_epoch_revision=NEW.lifecycle_revision
        AND ((r.cancel_epoch=NEW.cancel_epoch AND r.lifecycle_state IN ('work_accepted','delivering','delivery_blocked'))
          OR (r.cancel_epoch=NEW.cancel_epoch+1 AND r.lifecycle_state='cancelling' AND NEW.state='cancelled')))
    THEN RAISE(ABORT,'invalid_delivery_state_transition') END;
END;

CREATE TRIGGER delivery_attempt_event_capability_guard BEFORE INSERT ON orch_delivery_attempt_events BEGIN
  SELECT CASE WHEN orch_capability_ok('delivery_attempt_event',NEW.board_instance_id,NEW.tenant_scope,NEW.attempt_id,NEW.transition_seq,0,NEW.event_digest)!=1
    THEN RAISE(ABORT,'missing_delivery_attempt_event_capability') END;
END;

CREATE TRIGGER reconcile_queue_update_guard BEFORE UPDATE ON orch_reconcile_queue BEGIN
  SELECT CASE WHEN NEW.board_instance_id!=OLD.board_instance_id OR NEW.tenant_scope!=OLD.tenant_scope
    OR NEW.event_id!=OLD.event_id OR NEW.consumer_kind!=OLD.consumer_kind
    OR orch_capability_ok(
      CASE
        WHEN OLD.state='pending' AND NEW.state='claimed' THEN 'queue_claim'
        WHEN OLD.state='claimed' AND NEW.state='done' THEN 'queue_complete'
        WHEN NEW.state IN ('pending','dead_letter') THEN 'queue_recover'
        ELSE 'queue_invalid'
      END,
      NEW.board_instance_id,NEW.tenant_scope,CAST(NEW.event_id AS TEXT),
      NEW.claim_epoch,0,NEW.consumer_kind
    )!=1
    OR NOT ((OLD.state='pending' AND NEW.state IN ('claimed','dead_letter'))
      OR (OLD.state='claimed' AND NEW.state IN ('pending','done','dead_letter')))
    OR (NEW.state='claimed' AND (NEW.claim_owner IS NULL OR NEW.claim_token_hash IS NULL
      OR NEW.claim_epoch!=OLD.claim_epoch+1 OR NEW.claim_expires_at<=unixepoch('now')
      OR NEW.attempts!=OLD.attempts+1))
    OR (NEW.state!='claimed' AND (NEW.claim_owner IS NOT NULL OR NEW.claim_token_hash IS NOT NULL OR NEW.claim_expires_at IS NOT NULL))
    OR (NEW.state='done' AND NOT EXISTS (SELECT 1 FROM orch_effect_ledger e
      WHERE e.board_instance_id=NEW.board_instance_id AND e.tenant_scope=NEW.tenant_scope
        AND e.event_id=NEW.event_id AND e.consumer_kind=NEW.consumer_kind
        AND e.effect_digest=NEW.done_effect_digest))
    OR (NEW.state='dead_letter' AND OLD.state='claimed' AND OLD.claim_expires_at>unixepoch('now'))
    THEN RAISE(ABORT,'invalid_reconcile_queue_transition') END;
END;
CREATE TRIGGER reconcile_queue_delete_guard BEFORE DELETE ON orch_reconcile_queue BEGIN SELECT RAISE(ABORT,'reconcile_queue_delete_forbidden'); END;
CREATE TRIGGER effect_ledger_update_guard BEFORE UPDATE ON orch_effect_ledger BEGIN SELECT RAISE(ABORT,'immutable_effect_ledger'); END;
CREATE TRIGGER effect_ledger_delete_guard BEFORE DELETE ON orch_effect_ledger BEGIN SELECT RAISE(ABORT,'effect_ledger_delete_forbidden'); END;

CREATE TRIGGER resend_authorization_insert_guard BEFORE INSERT ON orch_delivery_resend_authorizations BEGIN
  SELECT CASE WHEN orch_capability_ok('resend_authorize',NEW.board_instance_id,NEW.tenant_scope,NEW.obligation_id,NEW.lifecycle_revision,NEW.cancel_epoch,NEW.capability_digest)!=1
    OR NEW.expires_at<=unixepoch('now') OR NEW.consumed_at IS NOT NULL
    OR NOT EXISTS (SELECT 1 FROM orch_delivery_obligations d
      WHERE d.board_instance_id=NEW.board_instance_id AND d.tenant_scope=NEW.tenant_scope
        AND d.obligation_id=NEW.obligation_id AND d.state='unknown'
        AND d.unknown_attempt_id=NEW.unknown_attempt_id
        AND d.lifecycle_revision=NEW.lifecycle_revision AND d.cancel_epoch=NEW.cancel_epoch)
    THEN RAISE(ABORT,'invalid_resend_authorization') END;
END;
CREATE TRIGGER resend_authorization_update_guard BEFORE UPDATE ON orch_delivery_resend_authorizations BEGIN
  SELECT CASE WHEN NEW.board_instance_id!=OLD.board_instance_id OR NEW.tenant_scope!=OLD.tenant_scope
    OR NEW.authorization_id!=OLD.authorization_id OR NEW.obligation_id!=OLD.obligation_id
    OR NEW.unknown_attempt_id!=OLD.unknown_attempt_id OR NEW.lifecycle_revision!=OLD.lifecycle_revision
    OR NEW.cancel_epoch!=OLD.cancel_epoch OR NEW.capability_digest!=OLD.capability_digest
    OR NEW.authorized_by!=OLD.authorized_by OR NEW.expires_at!=OLD.expires_at OR NEW.created_at!=OLD.created_at
    OR OLD.consumed_at IS NOT NULL OR NEW.consumed_at IS NULL
    OR orch_capability_ok('resend_consume',NEW.board_instance_id,NEW.tenant_scope,NEW.obligation_id,NEW.lifecycle_revision,NEW.cancel_epoch,NEW.authorization_id)!=1
    THEN RAISE(ABORT,'invalid_resend_authorization_consume') END;
END;
CREATE TRIGGER resend_authorization_delete_guard BEFORE DELETE ON orch_delivery_resend_authorizations BEGIN SELECT RAISE(ABORT,'resend_authorization_delete_forbidden'); END;

CREATE TRIGGER mutation_log_insert_guard BEFORE INSERT ON orch_mutation_log BEGIN
  SELECT CASE WHEN orch_capability_ok('mutation_log',NEW.board_instance_id,NEW.tenant_scope,NEW.orch_id,NEW.lifecycle_revision,NEW.cancel_epoch,NEW.mutation_digest)!=1
    THEN RAISE(ABORT,'missing_mutation_log_capability') END;
END;

WITH RECURSIVE
seed(task_id) AS (
  SELECT parent_task_id FROM orch_requests
   WHERE board_instance_id=:board AND tenant_scope=:tenant AND orch_id=:orch
  UNION
  SELECT task_id FROM orch_nodes
   WHERE board_instance_id=:board AND tenant_scope=:tenant AND orch_id=:orch
  UNION
  SELECT id FROM tasks
   WHERE orch_board_instance_id=:board AND orch_tenant_scope=:tenant AND orch_id=:orch
  UNION
  SELECT parent_task_id FROM orch_external_edges
   WHERE board_instance_id=:board AND tenant_scope=:tenant AND orch_id=:orch
  UNION
  SELECT child_task_id FROM orch_external_edges
   WHERE board_instance_id=:board AND tenant_scope=:tenant AND orch_id=:orch
),
closure(task_id) AS (
  SELECT task_id FROM seed
  UNION
  SELECT l.child_id FROM task_links l JOIN closure c ON l.parent_id=c.task_id
  UNION
  SELECT l.parent_id FROM task_links l JOIN closure c ON l.child_id=c.task_id
)
SELECT task_id FROM closure;