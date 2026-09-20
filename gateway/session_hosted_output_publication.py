"""Canonical terminal publication with explicit Files custody and final fences.

Port of the #99159 publication order, keeping the canonical planner/append path
and limiting byte transport to the coordinator's own store for this increment.
"""

from collections.abc import Mapping

from gateway import hosted_room_discussion as discussion, hosted_rooms
from gateway.hosted_room_artifacts import RoomArtifactOutbox, RoomArtifactScope
from hermes_state_runtime import RuntimeStoreError
from tui_gateway.hosted_room_artifact_service import prepare_output, acknowledge_published


class CanonicalHostedOutputPublisher:
    @property
    def output_attachments(self):
        return self.attachments

    def _output_source(self, room, task):
        result = task.get("result")
        if not isinstance(result, Mapping) or not result.get("artifacts"):
            return None
        scope = RoomArtifactScope.from_mapping(result.get("artifact_scope") or {})
        if scope.target_install_id != scope.home_install_id:
            from gateway.session_peer_output_custody import PeerOutputCustody
            source = PeerOutputCustody(self, scope, result["artifacts"], result, task["cancel_generation"])
            source._route(scope)
            return scope, result["artifacts"], source
        if (scope.home_install_id != room["authority_gateway_id"]
                or scope.target_install_id != scope.home_install_id
                or self.profile_homes().get(scope.target_profile) != self.root):
            raise RuntimeStoreError("unsupported_output_route")
        outbox = RoomArtifactOutbox(self.db_path)
        return scope, result["artifacts"], outbox

    def _acknowledge_output(self, scope, manifest, outbox):
        def acknowledge_or_retired(scope, artifact_ids, *, message_event_id):
            # Called only after exact canonical event and byte verification.
            # A durable fence with no pending reclamation survives ACK expiry;
            # missing receipt rows without that evidence still require ACK.
            if outbox.retirement_complete(scope):
                return 0
            return outbox.acknowledge(scope, artifact_ids, message_event_id=message_event_id)

        return acknowledge_published(self.output_attachments, scope=scope, manifest=manifest,
                                      acknowledge=acknowledge_or_retired)

    def _publish_terminal_tasks(self, room: Mapping) -> bool:
        changed, room_id, local_profiles = False, str(room["room_id"]), self.local_profiles()
        cursor = int(room["latest_seq"])
        for task in self._list_tasks(room_id, ("deferred", "settled", "failed", "cancelled")):
            status, generation = task["status"], int(task["execution_generation"])
            output = self._output_source(room, task)
            if self.policy_checkpoint.publication_exists(
                    room_id=room_id, task_id=task["identity"].task_id, status=status, execution_generation=generation):
                if output is not None:
                    scope, manifest, outbox = output
                    # Positive publication is required for source ACK. A
                    # cancelled/silent old reply has no visible file to ACK.
                    event_id = "dmessage:" + scope.task_id.removeprefix("dtask:")
                    if any(e["event_id"] == event_id for e in self._events(room_id)):
                        self._acknowledge_output(scope, manifest, outbox)
                    else:
                        outbox.discard_durably(scope)
                continue
            task_events = self.policy_checkpoint.events_for_task(
                room_id=room_id, source_event_seq=int(task["payload"]["source_event_seq"]),
                input_context=task["payload"].get("input_context"), task_id=task["identity"].task_id)
            plan = discussion.reconstruct_task_plan(room, task_events, task, local_profiles=local_profiles)
            message_id = "dmessage:" + task["identity"].task_id.removeprefix("dtask:")
            existing_message = next((e for e in task_events
                                     if e["event_id"] == message_id and e["kind"] == "message.member"), None)
            if existing_message:
                task_events = [e for e in task_events if e["kind"] != "message.user"
                               or int(e["seq"]) <= int(task["payload"]["source_event_seq"])]
            result, expected_output = task.get("result"), None
            # A superseded peer result has no visible file to import or ACK.
            # Determine that before byte reads so a lost discard reply can replay
            # even after the target has physically removed its private output.
            initial = discussion.plan_publication(
                room, task_events, plan, status=status, result=result,
                execution_generation=generation if status == "deferred" else None, local_profiles=local_profiles)
            peer_discard = (output is not None and output[0].target_install_id != output[0].home_install_id
                            and initial.terminal_kind in {"turn.cancelled", "turn.failed"})
            if output is not None:
                scope, manifest, outbox = output
                expected_output = dict(scope=scope.as_mapping(), manifest=manifest,
                                       cancel_generation=task["cancel_generation"])
                if scope.target_install_id != scope.home_install_id:
                    expected_output["peer_custody"] = outbox
                if peer_discard:
                    self.output_attachments.abort_unpublished_event(room_id=room_id, event_id=message_id)
                    outbox.discard_durably(scope)
                    attachments = []
                elif existing_message:
                    attachments = existing_message["payload"]["attachments"]
                else:
                    prepared = prepare_output(self.output_attachments, scope=scope, manifest=manifest,
                        recipient_member_ids=task["payload"].get("recipient_member_ids"), read_artifact=outbox.read)
                    attachments = prepared["payload"]["attachments"]
                result = {**result, "attachments": attachments,
                          "recipient_member_ids": task["payload"].get("recipient_member_ids")}
            publication = discussion.plan_publication(
                room, task_events, plan, status=status, result=result,
                execution_generation=generation if status == "deferred" else None, local_profiles=local_profiles)
            try:
                for event in publication.events:
                    appended = hosted_rooms.append_event(self.db_path, **event.append_kwargs(room_id),
                        expected_latest_seq=cursor, **({"expected_output": expected_output} if expected_output else {}))
                    cursor = max(cursor, int(appended["seq"]))
            except Exception:
                if output is not None:
                    # The store checks the journal in its own write transaction:
                    # a competing committed event can never lose its bytes.
                    self.output_attachments.abort_unpublished_event(room_id=room_id, event_id=message_id)
                raise
            if output is not None:
                if any(e.kind == "message.member" for e in publication.events):
                    self._acknowledge_output(scope, manifest, outbox)
                elif not peer_discard:
                    self.output_attachments.abort_unpublished_event(room_id=room_id, event_id=message_id)
                    outbox.discard_durably(scope)
            changed = True
        return changed
