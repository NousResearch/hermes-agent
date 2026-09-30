# Java Test Template (JUnit 5 + Mockito)

```java
package com.example.test;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

public class ExampleTest {

    @Mock
    private Dependency dependency;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testMethod() {
        // Given
        when(dependency.getValue()).thenReturn("expected");

        // When
        String result = sut.method();

        // Then
        assertEquals("expected", result);
        verify(dependency).getValue();
    }
}
```

## Patterns

- Use JUnit 5 (org.junit.jupiter.api)
- Use Mockito for mocks
- Given-When-Then pattern
- Verify all mock interactions
