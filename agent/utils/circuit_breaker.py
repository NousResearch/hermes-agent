import time
import logging

logger = logging.getLogger("hermes.circuit_breaker")

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_state_change = time.time()

    def allow_request(self) -> bool:
        now = time.time()
        if self.state == "OPEN":
            if now - self.last_state_change > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.last_state_change = now
                logger.info("Circuit breaker entered HALF_OPEN state, allowing a trial request")
                return True
            return False
        return True

    def record_success(self):
        if self.state != "CLOSED":
            logger.info("Circuit breaker recovered! Changing state from %s to CLOSED", self.state)
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_state_change = time.time()

    def record_failure(self):
        self.failure_count += 1
        now = time.time()
        if self.failure_count >= self.failure_threshold and self.state != "OPEN":
            self.state = "OPEN"
            self.last_state_change = now
            logger.warning(
                "Circuit breaker tripped! State changed to OPEN for %.1fs. Failure count: %d",
                self.recovery_timeout,
                self.failure_count,
            )
