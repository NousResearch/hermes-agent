"""Canonical terminal publication with explicit Files custody and final fences.

Port of the #99159 publication order, keeping the canonical planner/append path
and limiting byte transport to the coordinator's own store for this increment.
"""

from collections.abc import Mapping

from gateway import hosted_room_discussion as discussion, hosted_rooms
from gateway.hosted_room_artifacts import RoomArtifactError, RoomArtifactOutbox, RoomArtifactScope
from hermes_state_runtime import RuntimeStoreError
from gateway.session_hosted_output_retry import CanonicalOutputRetry, retryable
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_artifact_service import prepare_output, acknowledge_published


class CanonicalHostedOutputPublisher(CanonicalOutputRetry):
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

    def _prepare_terminal_tasks(self, room):
        return self._publish_terminal_tasks(room, defer_errors=True)

    def _publish_terminal_tasks(self, room: Mapping, *, defer_errors=False) -> bool:
        # Serialization is per publisher, never the policy/owner lock across I/O.
        # Direct callers retain exception reporting; scheduler callers consume the
        # durable disposition after every independent sibling has had a chance.
        with self._output_publish_lock:
            self._prune_output_retry_metadata(str(room['room_id']))
            self._unblock_authenticated_output_routes(str(room['room_id']))
            changed, errors = False, []
            for task in self._list_tasks(str(room["room_id"]), ("deferred", "settled", "failed", "cancelled")):
                metadata, progress = None, ['publish']
                has_output = isinstance(task.get('result'), Mapping) and bool(task['result'].get('artifacts'))
                try:
                    if has_output:
                        metadata = self._begin_output_retry(task)
                        if metadata is None:
                            continue
                        if 'completed_operation' in metadata:
                            completed, metadata = metadata, None
                            self._recheck_output_completion(task, completed)
                            continue
                    current = self._room(str(room['room_id']))
                    changed = self._publish_one_output(current, task, progress) or changed
                except Exception as exc:
                    if metadata is None:
                        errors.append(exc)
                        continue
                    self._finish_output_retry(task, metadata, operation=progress[0], error=exc)
                    if not defer_errors or not (retryable(exc) or isinstance(exc, PeerRunsHTTPError)):
                        errors.append(exc)
                else:
                    if metadata is not None:
                        try:
                            self._finish_output_retry(task, metadata, operation=progress[0])
                        except Exception as exc:
                            # A failed owner completion commit leaves the pre-I/O
                            # retry reservation intact. It is not remote denial.
                            errors.append(exc)
            if errors:
                raise errors[0]
            return changed

    def _recheck_output_completion(self, task, completed):
        # Suppress network, not local publication/byte integrity checks. Removed
        # producer bytes are never read: ACK evidence belongs to Home's copy.
        if completed['completed_operation'] == 'ack':
            result = task['result']
            acknowledge_published(self.output_attachments,
                scope=RoomArtifactScope.from_mapping(result['artifact_scope']), manifest=result['artifacts'],
                acknowledge=lambda *args, **kwargs: None)
        with self.authority.db._read_ctx() as conn:
            current = self._output_metadata(conn, self._output_key(task))
            if (any(current[k] != completed[k] for k in ('work', 'route', 'lineage', 'member_id'))
                    or float(self._artifact_clock()) >= completed['valid_until']):
                raise RoomArtifactError('Group Chat output completion changed')

    def _publish_one_output(self, room, task, progress):
        room_id, local_profiles = str(room['room_id']), self.local_profiles()
        cursor = int(room['latest_seq'])
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
                    progress[0] = 'ack'
                    self._acknowledge_output(scope, manifest, outbox)
                else:
                    progress[0] = 'discard'
                    outbox.discard_durably(scope)
            return False
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
                progress[0] = 'discard'
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
                progress[0] = 'ack'
                self._acknowledge_output(scope, manifest, outbox)
            elif not peer_discard:
                self.output_attachments.abort_unpublished_event(room_id=room_id, event_id=message_id)
                progress[0] = 'discard'
                outbox.discard_durably(scope)
        return True
