import json, os, subprocess, sys, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "hooks"))
import plan_guard as pg, deploy_state as ds, classify_bedrock_error as cb, command_guard as cg


def rc(addr, rtype, actions, after=None, replace_paths=None):
    ch = {"actions": actions, "after": after}
    if replace_paths: ch["replace_paths"] = replace_paths
    return {"address": addr, "type": rtype, "mode": "managed", "change": ch}


class PlanGuard(unittest.TestCase):
    def ev(self, *rcs, **kw): return pg.evaluate({"resource_changes": list(rcs)}, **kw)

    def test_tag_change_safe(self):
        r = self.ev(rc("aws_s3_bucket_policy.x", "aws_s3_bucket_policy", ["no-op"]),
                    rc("aws_cloudwatch_metric_alarm.a", "aws_cloudwatch_metric_alarm", ["update"]))
        self.assertEqual(r["verdict"], "SAFE"); self.assertEqual(r["counts"]["change"], 1)

    def test_instance_replace_blocked_and_counted(self):
        r = self.ev(rc("aws_instance.runtime", "aws_instance", ["delete", "create"], replace_paths=[["ami"]]))
        self.assertEqual(r["verdict"], "BLOCKED"); self.assertEqual(r["counts"]["replace"], 1)
        self.assertIn("ami", r["findings"][0]["reason"])

    def test_create_before_destroy_is_replace(self):
        self.assertEqual(pg.classify_actions(["create", "delete"]), "replace")

    def test_approved_replace_is_review(self):
        r = self.ev(rc("aws_instance.runtime", "aws_instance", ["delete", "create"]),
                    allow_replace=["aws_instance.runtime"])
        self.assertEqual(r["verdict"], "REVIEW")

    def test_ebs_destroy_blocked(self):
        self.assertEqual(self.ev(rc("aws_ebs_volume.state", "aws_ebs_volume", ["delete"]))["verdict"], "BLOCKED")

    def test_iam_wildcard_blocked(self):
        pol = json.dumps({"Statement": [{"Effect": "Allow", "Action": ["bedrock:InvokeModel"], "Resource": "*"}]})
        r = self.ev(rc("aws_iam_role_policy.b", "aws_iam_role_policy", ["update"], after={"policy": pol}))
        self.assertEqual(r["verdict"], "BLOCKED")

    def test_iam_scoped_arn_is_review_not_blocked(self):
        pol = json.dumps({"Statement": [{"Effect": "Allow",
            "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            "Resource": ["arn:aws:bedrock:eu-west-2:111122223333:inference-profile/eu.anthropic.claude-sonnet-4-6",
                         "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-6"]}]})
        r = self.ev(rc("aws_iam_role_policy.b", "aws_iam_role_policy", ["update"], after={"policy": pol}))
        self.assertEqual(r["verdict"], "REVIEW")

    def test_public_ssh_blocked(self):
        after = {"ingress": [{"from_port": 22, "to_port": 22, "cidr_blocks": ["0.0.0.0/0"]}]}
        self.assertEqual(self.ev(rc("aws_security_group.rt", "aws_security_group", ["update"], after=after))["verdict"], "BLOCKED")

    def test_data_sources_ignored(self):
        r = pg.evaluate({"resource_changes": [{"address": "data.x", "type": "aws_ami", "mode": "data", "change": {"actions": ["read"]}}]})
        self.assertEqual(r["verdict"], "SAFE")

    def test_cli_exit_codes(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"resource_changes": [rc("aws_ebs_volume.s", "aws_ebs_volume", ["delete"])]}, f)
        p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "plan_guard.py"), f.name], capture_output=True, text=True)
        self.assertEqual(p.returncode, 3); self.assertIn("VERDICT: BLOCKED", p.stdout)


class StateMachine(unittest.TestCase):
    def new(self):
        return {"env": "t", "state": None, "last_failure": None, "facts": {}, "decisions": [], "journal": []}

    def test_cannot_skip_review(self):
        s = self.new()
        for st in ["DISCOVER", "PREFLIGHT", "PLAN"]: ds.transition(s, st)
        with self.assertRaises(ds.TransitionError): ds.transition(s, "INFRASTRUCTURE_APPLY")

    def test_apply_needs_reviewed_sha(self):
        s = self.new()
        for st in ["DISCOVER", "PREFLIGHT", "PLAN", "PLAN_REVIEW"]: ds.transition(s, st)
        with self.assertRaises(ds.TransitionError): ds.transition(s, "INFRASTRUCTURE_APPLY")
        s["facts"]["reviewed_plan_sha256"] = "abc"; ds.transition(s, "INFRASTRUCTURE_APPLY")

    def test_new_plan_invalidates_review(self):
        s = self.new()
        for st in ["DISCOVER", "PREFLIGHT", "PLAN", "PLAN_REVIEW"]: ds.transition(s, st)
        s["facts"]["reviewed_plan_sha256"] = "abc"; ds.transition(s, "PLAN")
        self.assertNotIn("reviewed_plan_sha256", s["facts"])

    def test_failure_and_retry(self):
        s = self.new()
        for st in ["DISCOVER", "PREFLIGHT"]: ds.transition(s, st)
        ds.fail(s, "bedrock D"); self.assertEqual(s["state"], "FAILED_PREFLIGHT")
        with self.assertRaises(ds.TransitionError): ds.transition(s, "PLAN")
        ds.transition(s, "PREFLIGHT")

    def test_infra_failure_forces_replan(self):
        self.assertEqual(ds.allowed_next("FAILED_INFRASTRUCTURE"), ["PLAN"])

    def test_ready_requires_rollback(self):
        s = self.new(); s["state"] = "VERIFY"
        with self.assertRaises(ds.TransitionError): ds.transition(s, "READY")
        s["facts"]["rollback"] = {"image": "sha256:prev"}; ds.transition(s, "READY")

    def test_cli_roundtrip(self):
        d = tempfile.mkdtemp(); env = dict(os.environ, NOVA_STATE_FILE=os.path.join(d, "s.json"))
        run = lambda *a: subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "deploy_state.py"), *a], env=env, capture_output=True, text=True)
        self.assertEqual(run("init", "--env", "t").returncode, 0)
        self.assertEqual(run("advance", "DISCOVER").returncode, 0)
        self.assertEqual(run("advance", "READY").returncode, 3)
        self.assertIn('"state": "DISCOVER"', run("show").stdout)


class BedrockClassifier(unittest.TestCase):
    def cat(self, m): return cb.classify(m)["category"]
    def test_incident_is_D_not_schema(self):
        self.assertEqual(self.cat("An error occurred (ValidationException) when calling the Converse operation: Operation not allowed"), "D")
    def test_console(self): self.assertEqual(self.cat("Your account is not authorized to perform this action."), "D")
    def test_iam(self):
        self.assertEqual(self.cat("AccessDeniedException: User: arn:aws:sts::1:assumed-role/nova-test-runtime/i-1 is not authorized to perform: bedrock:InvokeModel"), "A")
    def test_profile_required(self):
        self.assertEqual(self.cat("ValidationException: Invocation of model ID anthropic.x with on-demand throughput isn't supported. Retry with an inference profile"), "C")
    def test_throttle_retryable(self):
        r = cb.classify("ThrottlingException: Too many requests"); self.assertEqual(r["category"], "G"); self.assertTrue(r["retryable"])
    def test_schema(self): self.assertEqual(self.cat("ValidationException: messages.0.content: field required"), "F")
    def test_unknown(self): self.assertEqual(self.cat("something odd"), "?")


class CommandGuard(unittest.TestCase):
    def b(self, c): return cg.decide_bash(c)[0]
    def test_destroy_denied(self):
        for c in ["terraform destroy", "docker run --rm hashicorp/terraform:1.16.3 destroy -auto-approve",
                  "terraform apply -destroy", "scripts/tf.sh destroy"]:
            self.assertEqual(self.b(c), "deny", c)
    def test_raw_apply_denied_tfsh_asks(self):
        self.assertEqual(self.b("terraform apply tfplan"), "deny")
        self.assertEqual(self.b("docker run --rm -v $PWD:/workspace hashicorp/terraform:1.16.3 apply tfplan"), "deny")
        self.assertEqual(self.b("NOVA_APPROVED_PLAN_SHA=abc scripts/tf.sh apply"), "ask")
    def test_chained_bypass_caught(self):
        self.assertEqual(self.b("scripts/tf.sh apply && aws ec2 terminate-instances --instance-ids i-1"), "deny")
    def test_plan_allowed(self):
        self.assertIsNone(self.b("scripts/tf.sh plan"))
        self.assertIsNone(self.b("terraform show -json tfplan"))
        self.assertIsNone(self.b("aws ec2 describe-instances --instance-ids i-1 --region eu-west-2"))
    def test_destructive_aws(self):
        for c in ["aws ec2 delete-volume --volume-id vol-1", "aws kms schedule-key-deletion --key-id k",
                  "aws s3 rm s3://b --recursive", "aws ec2 authorize-security-group-ingress --group-id sg --cidr 0.0.0.0/0 --port 22 --protocol tcp",
                  "cat ~/.aws/credentials"]:
            self.assertEqual(self.b(c), "deny", c)
    def test_prune_asks(self):
        self.assertEqual(self.b("aws ssm send-command --parameters 'commands=[\"python3 -m nova apply b.tgz --prune\"]'"), "ask")
    def test_edit_removing_prevent_destroy_asks(self):
        d, _ = cg.decide_edit("Edit", {"file_path": "main.tf", "old_string": "lifecycle { prevent_destroy = true }", "new_string": "lifecycle {}"})
        self.assertEqual(d, "ask")
    def test_edit_adding_star_asks(self):
        d, _ = cg.decide_edit("Edit", {"file_path": "iam.tf", "old_string": 'Resource = [local.arn]', "new_string": 'Resource = "*"'})
        self.assertEqual(d, "ask")
    def test_hook_io(self):
        p = subprocess.run([sys.executable, os.path.join(ROOT, "hooks", "command_guard.py")],
                           input=json.dumps({"tool_name": "Bash", "tool_input": {"command": "terraform destroy"}}),
                           capture_output=True, text=True)
        self.assertEqual(json.loads(p.stdout)["hookSpecificOutput"]["permissionDecision"], "deny")


if __name__ == "__main__":
    unittest.main()
